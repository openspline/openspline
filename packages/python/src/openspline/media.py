"""Optional native media receiver for framework output adapters."""

import asyncio
import json

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription


class MediaReceiver:
    def __init__(self, avatar):
        self.avatar = avatar
        self.pc = RTCPeerConnection()
        self.frames = asyncio.Queue(100)
        self.tasks = set()
        self.clock = None

    async def start(self):
        response = await self.avatar.http.get(
            f"/v1/sessions/{self.avatar.id}/ice",
            headers={"Authorization": "Bearer " + self.avatar.session["token"]},
        )
        self.avatar._check(response)
        await self.pc.close()
        self.pc = RTCPeerConnection(
            RTCConfiguration(iceServers=[RTCIceServer(**s) for s in response.json()["iceServers"]])
        )
        self.pc.addTransceiver("audio", direction="recvonly")
        self.pc.addTransceiver("video", direction="recvonly")
        self.channel = self.pc.createDataChannel("openspline")

        @self.channel.on("open")
        def opened():
            self.channel.send(json.dumps({"type": "ready"}))

        @self.channel.on("message")
        def message(raw):
            event = json.loads(raw)
            if event["type"] == "clock":
                self.clock = event
            if event["type"] == "interrupted":
                self.clock = None
                while not self.frames.empty():
                    self.frames.get_nowait()

        @self.pc.on("track")
        def track(track):
            async def receive():
                while True:
                    frame = await track.recv()
                    await self.frames.put((track.kind, frame))

            task = asyncio.create_task(receive())
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

        await self.pc.setLocalDescription(await self.pc.createOffer())
        s = self.avatar.session
        response = await self.avatar.http.post(
            f"/v1/sessions/{self.avatar.id}/offer",
            headers={"Authorization": "Bearer " + s["token"]},
            json={"type": self.pc.localDescription.type, "sdp": self.pc.localDescription.sdp},
        )
        self.avatar._check(response)
        await self.pc.setRemoteDescription(RTCSessionDescription(**response.json()))

    def acknowledge(self):
        if self.clock and self.channel.readyState == "open":
            self.channel.send(
                json.dumps(
                    {
                        "type": "played",
                        "epoch": self.clock["epoch"],
                        "samples": self.clock["samples"],
                    }
                )
            )

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.frames.get()

    async def close(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.pc.close()
