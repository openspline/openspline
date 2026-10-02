"""Run with Pipecat's development WebRTC client at http://localhost:7860/client."""

import os

from openspline.integrations.pipecat import AvatarProcessor
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_transport import TransportParams


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(
        runner_args,
        {
            "webrtc": lambda: TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                video_out_enabled=True,
                video_out_width=512,
                video_out_height=512,
                vad_analyzer=SileroVADAnalyzer(),
            )
        },
    )
    llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"])
    context = LLMContext([{"role": "system", "content": "Be helpful and concise."}])
    aggregator = LLMContextAggregatorPair(context)
    pipeline = Pipeline(
        [
            transport.input(),
            OpenAISTTService(api_key=os.environ["OPENAI_API_KEY"]),
            aggregator.user(),
            llm,
            OpenAITTSService(api_key=os.environ["OPENAI_API_KEY"]),
            AvatarProcessor(os.environ["PORTRAIT"]),
            transport.output(),
            aggregator.assistant(),
        ]
    )
    task = PipelineTask(pipeline, params=PipelineParams(audio_out_sample_rate=48000))

    @transport.event_handler("on_client_connected")
    async def connected(transport, client):
        await task.queue_frame(
            LLMMessagesAppendFrame([{"role": "user", "content": "Say hello."}], run_llm=True)
        )

    @transport.event_handler("on_client_disconnected")
    async def disconnected(transport, client):
        await task.cancel()

    await PipelineRunner(handle_sigint=runner_args.handle_sigint).run(task)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
