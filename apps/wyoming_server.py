"""
VieNeu-TTS — Wyoming protocol server (for Home Assistant).
===========================================================
Streams 48 kHz audio to the client frame-by-frame as `Vieneu.infer_stream`
produces it (same native ONNX/CPU streaming path as `apps/web_stream.py`),
instead of buffering the whole utterance before sending anything.

`infer_stream` is a blocking, CPU-bound generator. The `wyoming` library runs
every connection as an `asyncio.Task` on ONE event loop (see `wyoming.server
.AsyncServer._handler_callback`), so iterating a blocking generator directly
inside `handle_event` would freeze that loop for the whole synthesis — no
other connection/event gets serviced meanwhile, which is exactly what causes
perceived delay/stalls with Home Assistant on long text. `web_stream.py`
avoids this for free because Starlette runs sync generators in a threadpool;
here we do the equivalent by hand: `infer_stream` runs in a worker thread and
pushes chunks into an `asyncio.Queue`, so the event loop stays free to write
each chunk to the socket the instant it's ready.

    uv run python -m apps.wyoming_server --uri tcp://0.0.0.0:10200
"""
import argparse
import asyncio
import logging
import socket
import threading
import time
from typing import Optional

import numpy as np

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize, SynthesizeStopped

from vieneu import Vieneu

_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 48_000
SAMPLE_WIDTH = 2
CHANNELS = 1

# Sentinel telling the async consumer the producer thread is done.
_STREAM_DONE = object()
# Bounded lookahead: caps memory and gives backpressure without serializing
# "compute next chunk" behind "send this chunk" (both proceed concurrently).
_QUEUE_MAXSIZE = 4


def _produce_audio(
    vieneu_instance: Vieneu,
    text: str,
    voice: Optional[str],
    queue: "asyncio.Queue",
    loop: asyncio.AbstractEventLoop,
    stop_event: threading.Event,
) -> None:
    """Run in a worker thread: pull chunks off the blocking engine generator
    and hand each one to the event loop as soon as it's produced."""
    try:
        for chunk in vieneu_instance.infer_stream(text, voice=voice):
            if stop_event.is_set():
                break
            if chunk is None or len(chunk) == 0:
                continue
            # queue.put() runs on the loop; .result() blocks *this* thread
            # (not the loop) until there's room, giving real backpressure.
            asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
    except Exception as exc:  # noqa: BLE001
        try:
            asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
        except Exception:
            pass
    finally:
        try:
            asyncio.run_coroutine_threadsafe(queue.put(_STREAM_DONE), loop).result()
        except Exception:
            pass


class VieNeuEventHandler(AsyncEventHandler):
    """Event handler for Wyoming protocol."""

    def __init__(self, cli_args, vieneu_instance, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.vieneu = vieneu_instance
        self._disable_nagle()

    def _disable_nagle(self) -> None:
        """TCP_NODELAY: without it, Nagle's algorithm can coalesce/delay the
        small, frequent audio-chunk writes streaming produces (no-op/harmless
        on unix sockets, which don't support the option)."""
        sock = self.writer.get_extra_info("socket")
        if sock is None:
            return
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except (OSError, AttributeError):
            pass

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.send_info()
            return True

        # SynthesizeStart/SynthesizeChunk/SynthesizeStop bracket HA's streaming
        # request, but HA always follows them with a full-text Synthesize event
        # "for backwards compatibility" — that's the one that actually drives
        # synthesis, so the bracketing events are no-ops here.
        if Synthesize.is_type(event.type):
            await self._handle_synthesize(Synthesize.from_event(event))
            return True

        return True

    async def _handle_synthesize(self, synthesize: Synthesize) -> None:
        text = synthesize.text
        voice = synthesize.voice.name if synthesize.voice else "Phạm Tuyên"
        _LOGGER.info(f"Synthesizing ({len(text)} chars): {text!r}")

        audio_started = False
        stop_event = threading.Event()
        queue: "asyncio.Queue" = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        loop = asyncio.get_running_loop()
        producer_future = loop.run_in_executor(
            None, _produce_audio, self.vieneu, text, voice, queue, loop, stop_event
        )

        t0 = time.perf_counter()
        first_at: Optional[float] = None
        n_chunks = 0
        emitted_samples = 0

        try:
            await self.write_event(
                AudioStart(rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS).event()
            )
            audio_started = True

            while True:
                item = await queue.get()
                if item is _STREAM_DONE:
                    break
                if isinstance(item, Exception):
                    raise item

                if first_at is None:
                    first_at = time.perf_counter() - t0
                    _LOGGER.info(f"⚡ TTFA (time-to-first-audio): {first_at * 1000:.0f} ms")

                chunk_int16 = (np.clip(item, -1.0, 1.0) * 32767).astype(np.int16)
                n_chunks += 1
                emitted_samples += len(item)
                await self.write_event(
                    AudioChunk(
                        audio=chunk_int16.tobytes(),
                        rate=SAMPLE_RATE,
                        width=SAMPLE_WIDTH,
                        channels=CHANNELS,
                    ).event()
                )

            if first_at is not None:
                gen_time = time.perf_counter() - t0
                audio_s = emitted_samples / SAMPLE_RATE
                rtf = gen_time / audio_s if audio_s else 0
                _LOGGER.info(
                    f"✅ {n_chunks} chunks | audio {audio_s:.2f}s | gen {gen_time:.2f}s"
                    + (f" | RTF {rtf:.3f} ({1 / rtf:.1f}x realtime)" if rtf else "")
                )
            else:
                _LOGGER.warning("No audio produced for this request")
        except Exception as e:
            _LOGGER.error(f"Error during synthesis: {e}", exc_info=True)
        finally:
            # Unblock the producer thread if it's parked on a full queue,
            # then wait for it to actually finish (avoids leaking threads).
            stop_event.set()
            while not queue.empty():
                queue.get_nowait()
            await producer_future

            if audio_started:
                try:
                    await self.write_event(AudioStop().event())
                    # HA's streaming TTS reader (_read_tts_audio) only stops
                    # waiting for events once it sees SynthesizeStopped — it
                    # does not treat AudioStop as end-of-stream.
                    await self.write_event(SynthesizeStopped().event())
                except Exception:
                    pass  # client likely disconnected mid-stream
            _LOGGER.info("Finished synthesis")

    async def send_info(self) -> None:
        """Send info about the TTS program."""
        voices = self.vieneu.list_preset_voices()
        tts_voices = []
        for label, voice_id in voices:
            tts_voices.append(
                TtsVoice(
                    name=voice_id,
                    description=label,
                    attribution=Attribution(name="VieNeu", url="https://github.com/mkbyme/tts-vietneu"),
                    installed=True,
                    version="3.0",
                    languages=["vi", "en"],
                )
            )

        program = TtsProgram(
            name="vieneu",
            description="VieNeu TTS Server",
            attribution=Attribution(name="VieNeu", url="https://github.com/mkbyme/tts-vietneu"),
            installed=True,
            version="3.0",
            voices=tts_voices,
            # Without this, Home Assistant's Wyoming TTS integration falls back
            # to async_get_tts_audio, which buffers the *entire* WAV before
            # playback starts — negating all of infer_stream's low-latency
            # chunking. Setting it True switches HA to async_stream_tts_audio,
            # which plays AudioChunks as they arrive.
            supports_synthesize_streaming=True,
        )

        await self.write_event(Info(tts=[program]).event())

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="tcp://0.0.0.0:10200", help="Unix domain socket or tcp uri")
    parser.add_argument("--gpu", action="store_true", help="Use PyTorch GPU backend instead of ONNX CPU")
    parser.add_argument(
        "--threads", type=int, default=0,
        help="ONNX/CPU intra-op threads (0 = auto: physical cores, capped at 8)",
    )
    parser.add_argument(
        "--precision", choices=["int8", "fp32"], default="int8",
        help="ONNX/CPU backbone precision: int8 (default, ~3x faster/4x smaller) or fp32 (max quality)",
    )
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    backend = "onnx"
    if args.gpu:
        backend = "pytorch"
    else:
        try:
            import torch
            if torch.cuda.is_available():
                backend = "pytorch"
        except ImportError:
            pass

    _LOGGER.info(f"Initializing Vieneu TTS with backend: {backend}")
    vieneu_kwargs = {"backend": backend}
    if backend == "onnx":
        vieneu_kwargs["threads"] = args.threads
        vieneu_kwargs["precision"] = args.precision
    vieneu_instance = Vieneu(**vieneu_kwargs)
    _LOGGER.info(
        f"TTS model loaded. backend={vieneu_instance.backend} "
        f"intra_op_threads={getattr(vieneu_instance.engine, 'ort_intra_op_threads', '?')}"
    )

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info(f"Ready. Listening on {args.uri}")

    try:
        await server.run(lambda *conn_args, **conn_kwargs: VieNeuEventHandler(args, vieneu_instance, *conn_args, **conn_kwargs))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    asyncio.run(main())
