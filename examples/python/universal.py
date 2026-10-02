import argparse
import asyncio

from openspline import Openspline


async def main(portrait, audio):
    async with Openspline().avatar(portrait) as avatar:
        print("Open this viewer before playback:", avatar.viewer_url, flush=True)
        await avatar.play_file(audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("portrait")
    parser.add_argument("audio")
    args = parser.parse_args()
    asyncio.run(main(args.portrait, args.audio))
