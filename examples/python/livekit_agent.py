import os

from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession
from livekit.plugins import openai
from openspline.integrations.livekit import AvatarSession

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    session = AgentSession(llm=openai.realtime.RealtimeModel())
    avatar = AvatarSession(os.environ["PORTRAIT"])
    await avatar.start(session, room=ctx.room)
    await session.start(room=ctx.room, agent=Agent(instructions="Be helpful and concise."))
    await session.generate_reply(instructions="Greet the user.")


if __name__ == "__main__":
    agents.cli.run_app(server)
