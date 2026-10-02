"""Local LiveKit transport contract test; no provider credentials required."""

import asyncio

from livekit import api, rtc
from livekit.agents.voice.avatar import DataStreamAudioOutput
from openspline import Openspline
from PIL import Image


async def main():
    image = "/tmp/openspline-livekit-portrait.png"
    Image.new("RGB", (512, 512), "green").save(image)
    room = rtc.Room()
    counts = {"audio": 0, "video": 0}
    tasks = []

    @room.on("track_subscribed")
    def subscribed(track, pub, participant):
        async def consume():
            stream = (
                rtc.AudioStream(track)
                if track.kind == rtc.TrackKind.KIND_AUDIO
                else rtc.VideoStream(track)
            )
            async for frame in stream:
                counts["audio" if track.kind == rtc.TrackKind.KIND_AUDIO else "video"] += 1

        tasks.append(asyncio.create_task(consume()))

    def token(identity):
        return (
            api.AccessToken("devkey", "secret")
            .with_identity(identity)
            .with_grants(api.VideoGrants(room_join=True, room="openspline-test"))
            .to_jwt()
        )

    await room.connect("ws://127.0.0.1:7880", token("agent"))
    try:
        async with Openspline(url="http://127.0.0.1:7861", api_key="local-test-key").avatar(
            image
        ) as avatar:
            response = await avatar.http.post(
                f"/v1/sessions/{avatar.id}/livekit",
                headers=avatar._headers(),
                json={
                    "url": "ws://127.0.0.1:7880",
                    "token": token("avatar"),
                    "sender_identity": "agent",
                },
            )
            avatar._check(response)
            output = DataStreamAudioOutput(
                room, destination_identity="avatar", wait_playback_start=True
            )
            try:
                for _ in range(20):
                    await output.capture_frame(rtc.AudioFrame(bytes(4800), 24000, 1, 2400))
                output.flush()
                await asyncio.wait_for(output.wait_for_playout(), 30)
                assert counts["audio"] and counts["video"], counts
                for _ in range(15):
                    await output.capture_frame(rtc.AudioFrame(bytes(4800), 24000, 1, 2400))
                output.clear_buffer()
                await asyncio.sleep(1)
                print(
                    {
                        "native_livekit": True,
                        "frames": counts,
                        "playback_acknowledged": True,
                        "interruption_exercised": True,
                    }
                )
            finally:
                pass
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await room.disconnect()


asyncio.run(main())
