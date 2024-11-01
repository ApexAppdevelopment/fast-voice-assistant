import os
import asyncio

from livekit.agents import JobContext, WorkerOptions, cli, JobProcess
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, silero, cartesia, openai

from dotenv import load_dotenv

load_dotenv()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Define the medical persona system prompt for Joy
    initial_ctx = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "You are Joy, a medical assistant persona. Your role is to assist Daisy with tasks related to Ms. "
                    "Epi Hospital. Keep responses clear, professional, and tailored to hospital protocols and medical needs. "
                    "This assistant was created by Livekit agents of AITEK PH Software under the leadership of Master E and Master ATP."
                ),
            )
        ]
    )

    assistant = VoiceAssistant(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(
            base_url="https://api.together.xyz/v1",
            api_key=os.environ.get("TOGETHER_API_KEY"),
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-lora",
        ),
        tts=cartesia.TTS(voice="248be419-c632-4f23-adf1-5324ed7dbf1d"),
        chat_ctx=initial_ctx,
    )

    await ctx.connect()
    assistant.start(ctx.room)
    await asyncio.sleep(1)
    await assistant.say("Hello, Daisy. How can I assist you with hospital tasks today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
