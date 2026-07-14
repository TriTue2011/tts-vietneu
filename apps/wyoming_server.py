import argparse
import asyncio
import logging
import numpy as np

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize

from vieneu import Vieneu

_LOGGER = logging.getLogger(__name__)

class VieNeuEventHandler(AsyncEventHandler):
    """Event handler for Wyoming protocol."""

    def __init__(self, cli_args, vieneu_instance, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.vieneu = vieneu_instance

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.send_info()
            return True

        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            _LOGGER.info(f"Synthesizing: {synthesize.text}")

            try:
                voice = synthesize.voice.name if synthesize.voice else "Phạm Tuyên"
                
                # Default sample rate for v3 turbo is 48000
                sample_rate = 48000
                sample_width = 2
                channels = 1

                await self.write_event(
                    AudioStart(
                        rate=sample_rate,
                        width=sample_width,
                        channels=channels,
                    ).event()
                )

                _LOGGER.debug("Starting audio inference stream")
                # Using infer_stream for streaming generation
                for chunk in self.vieneu.infer_stream(synthesize.text, voice=voice):
                    # chunk is np.float32 array, need to convert to 16-bit PCM
                    chunk_int16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
                    audio_bytes = chunk_int16.tobytes()
                    
                    await self.write_event(
                        AudioChunk(
                            audio=audio_bytes,
                            rate=sample_rate,
                            width=sample_width,
                            channels=channels,
                        ).event()
                    )

                await self.write_event(AudioStop().event())
                _LOGGER.info("Finished synthesis")
            except Exception as e:
                _LOGGER.error(f"Error during synthesis: {e}")

            return True

        return True

    async def send_info(self) -> None:
        """Send info about the TTS program."""
        voices = self.vieneu.list_preset_voices()
        tts_voices = []
        for label, voice_id in voices:
            tts_voices.append(
                TtsVoice(
                    name=voice_id,
                    description=label,
                    attribution=Attribution(name="VieNeu", url="https://github.com/pnnbao97/VieNeu-TTS"),
                    installed=True,
                    languages=["vi", "en"],
                )
            )

        program = TtsProgram(
            name="vieneu",
            description="VieNeu TTS Server",
            attribution=Attribution(name="VieNeu", url="https://github.com/pnnbao97/VieNeu-TTS"),
            installed=True,
            voices=tts_voices,
        )

        await self.write_event(Info(tts=[program]).event())

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="tcp://0.0.0.0:10200", help="Unix domain socket or tcp uri")
    parser.add_argument("--gpu", action="store_true", help="Use PyTorch GPU backend instead of ONNX CPU")
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
    vieneu_instance = Vieneu(backend=backend)
    _LOGGER.info("TTS Model loaded successfully. Starting server...")

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info(f"Ready. Listening on {args.uri}")

    try:
        await server.run(lambda *conn_args, **conn_kwargs: VieNeuEventHandler(args, vieneu_instance, *conn_args, **conn_kwargs))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    asyncio.run(main())
