"""TEN extension using its public AsyncExtension and AudioFrame interfaces."""

import json

from ten_runtime import AsyncExtension, CmdResult, Data, StatusCode

from openspline import Openspline


class AvatarExtension(AsyncExtension):
    async def on_start(self, env):
        raw, error = await env.get_property_to_json("")
        if error:
            raise ValueError("Cannot read openspline properties")
        config = json.loads(raw)
        self.avatar = (
            await Openspline().avatar(config["portrait"], config.get("quality", "low")).__aenter__()
        )
        descriptor = Data.create("avatar_session")
        descriptor.set_property_from_json("session", json.dumps(self.avatar.session))
        await env.send_data(descriptor)

    async def on_audio_frame(self, env, frame):
        await self.avatar.send_audio(
            bytes(frame.get_buf()),
            sample_rate=frame.get_sample_rate(),
            channels=frame.get_number_of_channels(),
        )

    async def on_data(self, env, data):
        if data.get_name() == "tts_audio_end":
            await self.avatar.end_turn(drain=False)
        elif data.get_name() == "flush":
            await self.avatar.interrupt()

    async def on_cmd(self, env, cmd):
        if cmd.get_name() == "flush":
            await self.avatar.interrupt()
        elif cmd.get_name() == "end_turn":
            await self.avatar.end_turn()
        await env.return_result(CmdResult.create(StatusCode.OK, cmd))

    async def on_stop(self, env):
        if getattr(self, "avatar", None):
            await self.avatar.close()
