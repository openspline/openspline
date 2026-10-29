"""Native LiveKit output. The GPU service publishes directly to the room."""

import asyncio
import contextlib
import math
import os
import secrets
from pathlib import Path
from urllib.parse import urlsplit

from livekit import api, rtc
from livekit.agents import AgentSession
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession
from livekit.agents.voice.avatar import DataStreamAudioOutput

from openspline import Openspline
from openspline.config import Quality


class AvatarSession(BaseAvatarSession):
    """Attach a portrait or saved Cloud avatar to an agent's connected room.

    Legacy ``url``, ``api_key`` and ``api_secret`` refer to LiveKit. Configure
    OpenSpline through ``client``. Sessions are single-use; close is idempotent.
    """

    def __init__(
        self,
        portrait: str | Path | bytes | None = None,
        quality: Quality | None = None,
        client: Openspline | None = None,
        url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        *,
        background: str | Path | bytes | None = None,
        avatar_id: str | None = None,
        join_timeout: float = 30.0,
    ) -> None:
        super().__init__()
        self.client = client or Openspline()
        # The SDK validates sources and quality without allocating a worker.
        self._source = self.client.avatar(
            portrait, quality, background=background, avatar_id=avatar_id
        )
        if (
            isinstance(join_timeout, bool)
            or not isinstance(join_timeout, (int, float))
            or not math.isfinite(join_timeout)
            or join_timeout <= 0
        ):
            raise ValueError("join_timeout must be a finite positive number of seconds")
        self.join_timeout = join_timeout
        self.url = url or os.environ.get("LIVEKIT_URL")
        self.key = api_key or os.environ.get("LIVEKIT_API_KEY")
        self.secret = api_secret or os.environ.get("LIVEKIT_API_SECRET")
        self.identity = "openspline-" + secrets.token_hex(6)
        self.avatar = None
        self._start_task = None
        self._close_task = None
        self._closed = False
        self._cleanup_lock = asyncio.Lock()
        self._output = None
        self._previous_output = None
        self._session = None

    @property
    def avatar_identity(self) -> str:
        return self.identity

    @property
    def provider(self) -> str:
        return "openspline"

    async def start(self, agent_session: AgentSession, room: rtc.Room) -> None:
        """Provision the worker and wait for video before allowing agent speech."""
        if self._closed or self._start_task is not None:
            raise RuntimeError("AvatarSession is single-use; create a new session")
        if not room.isconnected() or not room.name or not room.local_participant.identity:
            raise ValueError("Connect the agent to its room before starting the avatar")
        parsed = urlsplit(self.url or "")
        if (
            parsed.scheme not in {"ws", "wss"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Set LIVEKIT_URL to a ws:// or wss:// room server URL")
        if not self.key or not self.secret:
            raise ValueError("Set LIVEKIT_API_KEY and LIVEKIT_API_SECRET")
        self._start_task = asyncio.create_task(self._start(agent_session, room))
        await self._start_task

    async def _start(self, agent_session, room):
        try:
            await super().start(agent_session, room)
            self._session = agent_session
            token = (
                api.AccessToken(self.key, self.secret)
                .with_identity(self.identity)
                .with_kind("agent")
                .with_attributes({"lk.publish_on_behalf": room.local_participant.identity})
                .with_grants(api.VideoGrants(room_join=True, room=room.name))
                .to_jwt()
            )
            self.avatar = await self._source.__aenter__()
            await self.avatar.attach_livekit(
                url=self.url, token=token, sender_identity=room.local_participant.identity
            )
            await self.wait_for_join(timeout=self.join_timeout)
            self._previous_output = agent_session.output.audio
            self._output = DataStreamAudioOutput(
                room, destination_identity=self.identity, wait_playback_start=True
            )
            agent_session.output.audio = self._output
        except BaseException:
            self._closed = True
            # Cleanup must not replace a provisioning error or cancellation.
            with contextlib.suppress(Exception):
                await self._cleanup()
            raise

    async def _cleanup(self):
        async with self._cleanup_lock:
            avatar, self.avatar = self.avatar, None
            try:
                if self._session and self._output and self._session.output.audio is self._output:
                    self._session.output.audio = self._previous_output
            finally:
                self._output = self._previous_output = self._session = None
                try:
                    if avatar:
                        await avatar.close()
                finally:
                    await super().aclose()

    async def _close(self):
        if self._start_task and not self._start_task.done():
            self._start_task.cancel()
            await asyncio.gather(self._start_task, return_exceptions=True)
        await self._cleanup()

    async def aclose(self) -> None:
        """Release the worker, including when called during startup."""
        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)
