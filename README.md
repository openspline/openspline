<!-- <div align="center">
<a href="https://openspline.dev/">
  <img alt="Openspline : give your ai a face" src="https://github.com/openspline/openspline/blob/main/media/banner2.png?raw=true"  width="full">
</a>
</div> -->
# OpenSpline
<!-- <p align="center"> -->
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-2EA44F?style=flat-square"></a>
  <a href="#python-sdk"><img alt="Python 3.10 to 3.12" src="https://img.shields.io/badge/python-3.10%20to%203.12-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white"></a>
  <a href="#node-sdk"><img alt="Node.js 20+" src="https://img.shields.io/badge/node.js-20%2B-339933?style=flat-square&amp;logo=nodedotjs&amp;logoColor=white"></a>
  <a href="#gpu-workers"><img alt="NVIDIA GPU" src="https://img.shields.io/badge/GPU-NVIDIA-76B900?style=flat-square&amp;logo=nvidia&amp;logoColor=white"></a>
<!-- </p> -->



**Add a live face to any voice AI agent in five lines.** Turn a portrait and generated audio into synchronized video and speech on OpenSpline Cloud or your own GPU.


[Try Online (no GPU or installation needed)](https://dashboard.openspline.dev/explore)

[Read Docs](https://docs.openspline.dev/)

- **Bring your agent.** Keep your prompts, tools, microphone, and voice provider.
- **Python first, Node supported.** Stream audio or use a ready-made connector.
- **Ready to display.** Built-in viewer, React component, and web component.
- **Cloud or self-hosted.** Start with an openspline cloud, or run the models on your GPUs.

> **AI agents:** Download and attach [llms.md](docs/llms.md) for the complete documentation in one file.

## Examples

<!-- GitHub strips iframe/script embeds. These looping GIFs show published avatars. -->
<table>
  <tr>
    <th>Meera</th>
    <th>Acharya</th>
    <th>Lisa</th>
  </tr>
  <tr>
    <td align="center"><a href="https://dashboard.openspline.dev/meera"><img src="media/examples/meera.gif" alt="Meera talking avatar preview" width="240" height="240"></a><br><a href="https://dashboard.openspline.dev/meera">Try online</a></td>
    <td align="center"><a href="https://dashboard.openspline.dev/acharya-ji"><img src="media/examples/acharya.gif" alt="Acharya talking avatar preview" width="240" height="240"></a><br><a href="https://dashboard.openspline.dev/acharya-ji">Try online</a></td>
    <td align="center"><a href="https://dashboard.openspline.dev/hemant-virtual-testing2"><img src="media/examples/lisa.gif" alt="Lisa talking avatar preview" width="240" height="240"></a><br><a href="https://dashboard.openspline.dev/hemant-virtual-testing2">Try online</a></td>
  </tr>
  <tr>
    <th>Steve</th>
    <th>Emily</th>
    <th>Professor Nikolaev</th>
  </tr>
  <tr>
    <td align="center"><a href="https://dashboard.openspline.dev/steve"><img src="media/examples/steve.gif" alt="Steve talking avatar preview" width="240" height="240"></a><br><a href="https://dashboard.openspline.dev/steve">Try online</a></td>
    <td align="center"><a href="https://dashboard.openspline.dev/emily"><img src="media/examples/emily.gif" alt="Emily talking avatar preview" width="240" height="240"></a><br><a href="https://dashboard.openspline.dev/emily">Try online</a></td>
    <td align="center"><a href="https://dashboard.openspline.dev/professor-nikolaev"><img src="media/examples/professor.gif" alt="Professor Nikolaev talking avatar preview" width="240" height="240"></a><br><a href="https://dashboard.openspline.dev/professor-nikolaev">Try online</a></td>
  </tr>
</table>

# Quickstart

There are two ways to run avatar generation:

- [Fully Hosted Cloud](#openspline-cloud): If you don't have a GPU or want to scale.
- [Self Host](#self-host): Run on your own GPU locally.

**GPU compatibility:** PyTorch uses CUDA 12.8 for Blackwell (including RTX 50-series) and CUDA 12.6 for older NVIDIA GPUs. Pre-Ampere cards use FP32, which needs more VRAM and may be slower. AMD and Apple GPUs are not supported yet. Blackwell and pre-Ampere inference still need hardware validation.

## Openspline Cloud

Create and embed an avatar in the Cloud dashboard, or use the API to generate avatars on Openspline GPUs.

[Cloud Components & API Docs ↗](docs/cloud_quickstart.md)

## Self Host

### Install the GPU service

Run the following command in a linux device/WSL2 with NVIDIA GPU:

```bash
curl -fsSL https://raw.githubusercontent.com/openspline/openspline/main/install.sh | sh
```

The installer creates a private Python environment and downloads the Low and High models.

### Launch the demo

```bash
sh ~/.local/share/openspline/run.sh
```

Open [http://localhost:7860](http://localhost:7860) to access the demo.

### Python SDK

Install the client package where your agent runs:

```bash
pip install openspline
```

Stream generated audio to a portrait:

```python
from openspline import Openspline
client = Openspline()
async with client.avatar("portrait.jpg", quality="low") as avatar:
    print(avatar.viewer_url)
    await avatar.stream(agent.audio_stream(), sample_rate=24000)
```

[Add a custom photo or video background](docs/custom_background.md).

`stream()` accepts generated PCM audio and returns after playback finishes. WAV and MP3 files are also supported.

The GPU stays active throughout a connected session; smooth continuous motion requires inference to keep up with playback.

## Connectors

These snippets plug into an existing async agent. Openspline owns output playback; disable your previous speaker output to avoid playing audio twice. Provider credentials stay on your backend.

<details>
<summary>OpenAI Realtime</summary>

Install `pip install openai==2.54.0`. Wrap your existing OpenAI Realtime connection; tools and transcripts still reach your event handler.

```python
from openspline import Openspline
from openspline.integrations import OpenAIRealtime

async with Openspline().avatar("portrait.jpg") as avatar:
    print(avatar.viewer_url)
    await avatar.wait_for_viewer()
    adapter = OpenAIRealtime(avatar, connection)
    try:
        async for event in adapter.wrap(connection):
            await handle_event(event)  # your existing handler
    finally:
        adapter.close()
```

[Run the example](examples/python/openai_realtime.py): `OPENAI_API_KEY=... python examples/python/openai_realtime.py portrait.jpg`.

</details>

<details>
<summary>LiveKit (Python and Node)</summary>

The experimental LiveKit plugins support saved Cloud avatars and portrait uploads. Start the avatar before your existing agent session; your normal LiveKit frontend receives synchronized audio/video.

```python
from livekit.plugins import openspline

avatar = openspline.AvatarSession(avatar_id="YOUR_AVATAR_ID")
await avatar.start(agent_session, room=ctx.room)
```

```ts
import * as openspline from '@livekit/agents-plugin-openspline';

const avatar = new openspline.AvatarSession({ avatarId: 'YOUR_AVATAR_ID' });
await avatar.start(agentSession, ctx.room);
```

[Installation, complete examples, and recorded demos](docs/connectors/livekit.mdx) · [LiveKit integration guide](https://docs.livekit.io/agents/models/avatar/plugins/openspline/).

Set `OPENSPLINE_API_KEY` and your `LIVEKIT_*` credentials. For self-hosting, unset the Cloud key and use a `portrait` instead. Existing SDK integration imports remain supported.

</details>

<details>
<summary>Gemini Realtime</summary>

Install `pip install google-genai==1.75.0`. Forward every audio part from your existing Gemini Live session, including interruption and turn-end events.

```python
from openspline import Openspline
from openspline.integrations import GeminiLive

async with Openspline().avatar("portrait.jpg") as avatar:
    print(avatar.viewer_url)
    await avatar.wait_for_viewer()
    async for event in GeminiLive(avatar).wrap(live_session.receive()):
        await handle_event(event)  # your existing handler
    await avatar.end_turn()
```

[Run the example](examples/python/gemini.py): `GEMINI_API_KEY=... python examples/python/gemini.py portrait.jpg`. Repeat the receive loop for subsequent turns; set `GEMINI_LIVE_MODEL` to override the model.

</details>

<details>
<summary>ElevenLabs Agents</summary>

Connect an existing [ElevenLabs agent WebSocket](https://elevenlabs.io/docs/eleven-agents/api-reference/eleven-agents/websocket). No additional provider SDK is needed. Pass decoded events, including the initial metadata, so the adapter uses the agent's negotiated audio format.

```python
import json
from openspline import Openspline
from openspline.integrations import ElevenLabsAgents

async with Openspline().avatar("portrait.jpg") as avatar:
    print(avatar.viewer_url)
    await avatar.wait_for_viewer()
    adapter = ElevenLabsAgents(avatar)
    async for message in connection:
        event = json.loads(message)
        await adapter.handle(event)
        await handle_event(event)  # keep your input, pong, and tool handlers
```

The adapter supports PCM and μ-law audio, flushes final audio, and handles interruptions. It preserves transcripts and tool events for your existing handler. If attaching after connection setup, pass `audio_format="pcm_24000"` (or the negotiated format) to the constructor. Node exports `ElevenLabsAgents` from `@openspline/node/integrations`, with `await adapter.handle(event)` and an optional format as the second constructor argument.

[Python example](examples/python/elevenlabs_agents.py): `ELEVENLABS_AGENT_ID=... python examples/python/elevenlabs_agents.py portrait.jpg`. [Node example](examples/node/elevenlabs_agents.mjs): `ELEVENLABS_AGENT_ID=... node examples/node/elevenlabs_agents.mjs portrait.jpg`. Set `ELEVENLABS_API_KEY` for private agents; the examples request a signed URL on the backend. Each plays one response; the demo supports microphone conversations. Configure the agent's client events to include `audio`, `interruption`, and `agent_response_complete`. Custom client tools and required dynamic variables need your application's handlers and initiation data; the demo uses the agent's saved configuration.

</details>

<details>
<summary>Custom connector</summary>

Use any TTS provider or agent with an async PCM16 audio iterator. `stream()` buffers audio, flushes the final block, and waits for playback to finish.

```python
from openspline import Openspline
client = Openspline(quality="low")
async with client.avatar("portrait.jpg") as avatar:
    print(avatar.viewer_url)
    await avatar.stream(agent.audio_stream(), sample_rate=24000)
```

For callbacks, await `avatar.send_audio(chunk, sample_rate=24000)`, then `await avatar.end_turn()` when speech ends. Call `await avatar.interrupt()` on barge-in. Float32 uses `encoding="pcm_f32le"`; encoded streams use `encoding="mp3"`. [File example](examples/python/universal.py).

</details>

More connectors: [OpenAI Agents](examples/python/openai_agents.py), [Pipecat](examples/python/pipecat_bot.py), [Google ADK](examples/python/google_adk.py), [Vocode](examples/vocode/main.py), and [TEN](examples/ten/graph.json). Python examples have [locked dependencies](examples/python/requirements.lock); Vocode uses its [own environment](examples/vocode/requirements.lock).

## Configuration

Pass settings directly when initializing, or reuse a typed config object:

```python
from openspline import Openspline, OpensplineConfig

client = Openspline(url="http://localhost:7860", quality="low")
config = OpensplineConfig(quality="high", timeout=300, viewer_timeout=90)
client = Openspline(config=config)  # a dict works too
```

| Python / Node option | Default | Purpose |
| --- | --- | --- |
| `api_key` / `apiKey` | `OPENSPLINE_API_KEY` | Backend-only Cloud key; automatically selects Cloud |
| `url` | Cloud with a key; otherwise `http://localhost:7860` | Explicit URL or `OPENSPLINE_URL` overrides the service address |
| `quality` | `low` | Default quality: `low` or `high` |
| `timeout` | 180 s / 180,000 ms | HTTP and audio request timeout |
| `viewer_timeout` / `viewerTimeout` | 60 s / 60,000 ms | Wait for the playback destination |

Explicit Python parameters override the config object; explicit settings override environment defaults. `avatar(..., quality="high")` overrides that session only. Invalid configuration fails before connecting. [JSON Schema](config.schema.json) describes Python settings, with `#/$defs/node` and `#/$defs/server` for Node and worker configuration. Only `avatar.session` should be passed to a browser.

## Node SDK

Use Node 20+ and install the SDK with one command:

```bash
npm install @openspline/node
```

```ts
import { Openspline, type OpensplineConfig } from "@openspline/node";
const config: OpensplineConfig = { apiKey: "YOUR_API_KEY", quality: "low" };
const client = new Openspline(config);
await using avatar = await client.avatar("portrait.jpg");
console.log(avatar.viewerUrl);
await avatar.stream(agent.audioStream(), { sample_rate: 24000 });
```

[Add a custom photo or video background](docs/custom_background.md).

For a saved Cloud avatar, use `client.avatar({ avatarId: "YOUR_AVATAR_ID" })`. You can omit `apiKey` to read `OPENSPLINE_API_KEY` from your server environment. For a local service, omit the key and unset that environment variable.

[Node examples](examples/node): run `npm ci` there, then `node universal.mjs portrait.jpg speech.wav`. Without `await using`, call `avatar.close()` in `finally`.


## GPU workers

### Configure multiple workers

Edit `~/.local/share/openspline/workers.yaml` ([example](workers.yaml)). Each worker loads weights once and serves one session at a time; device assignments cannot overlap.

```yaml
workers:
  - { id: low-0, quality: low, devices: [0] }
  - { id: low-1, quality: low, devices: [1] }
```

The demo switches quality automatically for a single worker. For SDK sessions or multiple workers, configure a worker with `quality: high, devices: [0, 1]` to use a GPU pair, then run `sh ~/.local/share/openspline/run.sh --prepare` before starting. Busy workers return a typed `CapacityError`; an unconfigured quality returns `ConfigurationError`. IDs are relative to `CUDA_VISIBLE_DEVICES` when set. For multiple workers, remove `OPENSPLINE_GPU`, `OPENSPLINE_GPUS`, and `OPENSPLINE_QUALITY` overrides from the installed `.env` and configure devices in `workers.yaml`.

### Network settings

Set `OPENSPLINE_HOST`, `OPENSPLINE_PORT`, `OPENSPLINE_PUBLIC_URL`, and provider credentials in the installed `.env`. Use HTTPS for remote microphone access. WebRTC playback additionally needs reachable UDP ports or TURN configured in `ice_servers`. API reference: **http://localhost:7860/docs**. Health: `/readyz`; metrics: `/metrics`.

<details>
<summary>Development and validation</summary>

Run `uv sync --all-packages`, `npm ci && npm run build`, then `uv run pytest && npm test`. `uv run openspline serve --backend test --dev` starts a static-portrait development service without a GPU. Keep optional framework environments separate; from `examples/python`, use `uv pip sync requirements.lock` in an activated environment. Release workflows build and optionally publish Python and npm packages.

Regenerate the native installer dependency locks with `python scripts/lock-gpu.py` (requires `uv`). The installer combines `requirements/server.txt` with the selected `cu126.txt` or `cu128.txt` runtime lock.

Real low/high inference, GPU streaming, browser/React playback, and native LiveKit have been exercised. Short A100 tests measured about 64 FPS for low quality and 8 FPS for high quality under different load conditions; these are engine rates, not playback latency. Long-duration, multi-GPU, cloud-provider, and TEN runtime validation remain incomplete.

</details>

## License

[MIT](LICENSE). The vendored inference code retains its [Apache-2.0 license](packages/server/src/openspline_server/_vendor/LICENSE) and [third-party notices](packages/server/NOTICE); model weights have separate terms.

## Acknowledgments

Openspline builds on open research and infrastructure, with thanks to the teams and contributors behind:

- [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) for the audio-driven talking-head models that power avatar generation.
- [LiveKit Agents](https://github.com/livekit/agents) for the real-time agent framework that Openspline integrates with to bring avatars into voice conversations.
- [Wan](https://github.com/Wan-Video/Wan2.1) for video generation research and VAE components used in the inference backend through SoulX-FlashHead.
- [LTX-Video](https://github.com/Lightricks/LTX-Video) for video autoencoder components used in the inference backend through SoulX-FlashHead.

Openspline carries this work forward with a self-hosted runtime for interactive avatars: synchronized audio and video streaming, continuous idle motion, interruption handling, GPU worker orchestration, and a shared API across voice providers. Its Python and Node SDKs, agent connectors, and embeddable players make these capabilities available to developers building their own conversational experiences.
