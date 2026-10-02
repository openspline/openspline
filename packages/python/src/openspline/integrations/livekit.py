"""LiveKit native avatar plugin. Install the server's LiveKit extra as well."""

import os
import secrets

from livekit import api
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession
from livekit.agents.voice.avatar import DataStreamAudioOutput

from openspline import Openspline


class AvatarSession(BaseAvatarSession):
    def __init__(
        self, portrait, quality=None, client=None, url=None, api_key=None, api_secret=None
    ):
        super().__init__()
        self.client = client or Openspline()
        self.portrait = portrait
        self.quality = quality
        self.url = url or os.environ.get("LIVEKIT_URL")
        self.key = api_key or os.environ.get("LIVEKIT_API_KEY")
        self.secret = api_secret or os.environ.get("LIVEKIT_API_SECRET")
        self.identity = "openspline-" + secrets.token_hex(6)
        self.avatar = None

    @property
    def avatar_identity(self):
        return self.identity

    @property
    def provider(self):
        return "openspline"

    async def start(self, agent_session, room):
        await super().start(agent_session, room)
        self.avatar = await self.client.avatar(self.portrait, self.quality).__aenter__()
        token = (
            api.AccessToken(self.key, self.secret)
            .with_identity(self.identity)
            .with_kind("agent")
            .with_attributes({"lk.publish_on_behalf": room.local_participant.identity})
            .with_grants(api.VideoGrants(room_join=True, room=room.name))
            .to_jwt()
        )
        try:
            response = await self.avatar.http.post(
                f"/v1/sessions/{self.avatar.id}/livekit",
                headers=self.avatar._headers(),
                json={
                    "url": self.url,
                    "token": token,
                    "sender_identity": room.local_participant.identity,
                },
            )
            self.avatar._check(response)
            agent_session.output.audio = DataStreamAudioOutput(
                room, destination_identity=self.identity, wait_playback_start=True
            )
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self):
        if self.avatar:
            await self.avatar.close()
            self.avatar = None
        await super().aclose()
