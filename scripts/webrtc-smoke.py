"""Exercise the actual HTTP, SDK, WebRTC, playout feedback and release path."""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
from openspline import CapacityError, Openspline
from PIL import Image


async def run(args):
    client = Openspline(timeout=300)
    if not args.portrait:
        portrait = Path(os.getenv("TMPDIR", "/tmp")) / "openspline-test-portrait.png"
        Image.new("RGB", (512, 512), "#aac478").save(portrait)
    else:
        portrait = Path(args.portrait)
    result = {
        "quality": args.quality,
        "audio_frames": 0,
        "video_frames": 0,
        "interruptions": 0,
        "capacity_rejected": False,
    }
    started = time.monotonic()
    async with client.avatar(portrait, quality=args.quality) as avatar:
        result["prepare_seconds"] = time.monotonic() - started
        try:
            async with client.avatar(portrait, quality=args.quality):
                pass
        except CapacityError:
            result["capacity_rejected"] = True
        receiver = await avatar.media()

        async def output():
            async for kind, frame in receiver:
                result[kind + "_frames"] += 1
                if kind == "audio":
                    receiver.acknowledge()

        task = asyncio.create_task(output())
        await avatar.wait_for_viewer()
        try:

            async def check_idle():
                before = result["video_frames"]
                await asyncio.sleep(1.5)
                assert result["video_frames"] > before
                response = await avatar.http.get("/metrics")
                metric = f'openspline_idle_blocks{{session="{avatar.id}"}} '
                blocks = next(
                    float(line[len(metric) :])
                    for line in response.text.splitlines()
                    if line.startswith(metric)
                )
                assert blocks > 0
                return blocks

            result["initial_idle_blocks"] = await check_idle()
            deadline = time.monotonic() + args.seconds
            turn = 0
            while time.monotonic() < deadline:
                for _ in range(20):
                    samples = (np.sin(np.arange(2400) * 2 * np.pi * 200 / 24000) * 3000).astype(
                        "<i2"
                    )
                    await avatar.send_audio(samples.tobytes())
                    await asyncio.sleep(0.1)
                if turn % 5 == 3:
                    before = time.monotonic()
                    await avatar.interrupt()
                    result["interruptions"] += 1
                    result["interrupt_ack_seconds"] = time.monotonic() - before
                else:
                    await avatar.end_turn()
                turn += 1
            result["turns"] = turn
            await avatar.interrupt()
            result["interruptions"] += 1
            result["idle_blocks_after_interrupt"] = await check_idle()
            assert result["idle_blocks_after_interrupt"] > result["initial_idle_blocks"]
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    result["elapsed_seconds"] = time.monotonic() - started
    # A fresh session must acquire the same released worker.
    async with client.avatar(portrait, quality=args.quality):
        result["worker_reused"] = True
    assert result["capacity_rejected"] and result["worker_reused"]
    assert result["audio_frames"] > 0 and result["video_frames"] > 0
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--portrait")
    p.add_argument("--quality", choices=["low", "high"], default="low")
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--output")
    asyncio.run(run(p.parse_args()))
