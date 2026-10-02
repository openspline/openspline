# openspline

**Add a live face to any voice AI agent in five lines.** Turn a portrait and generated audio into synchronized video and speech on your own GPU.

[Quick start](#quick-start) · [Connectors](#connectors) · [Configuration](#configuration) · [Frontend](#frontend)

- **Bring your agent.** Keep your prompts, tools, microphone, and voice provider.
- **Python first, Node supported.** Stream audio or use a ready-made connector.
- **Ready to display.** Built-in viewer, React component, and web component.
- **Self-hosted.** One active session per GPU worker; add GPUs for concurrency.

## Quick start

Start the GPU service on Linux with Docker and NVIDIA Container Toolkit:

```bash
curl -fsSL https://raw.githubusercontent.com/openspline/openspline/main/install.sh | sh
```

This creates a service key, downloads the low-quality model, and starts openspline at **http://localhost:7860**. Existing configuration is preserved when the command is run again. Set `OPENSPLINE_VERSION=v0.1.0` before the command to install a release, or `OPENSPLINE_INSTALL_DIR` to choose a directory.

Install the Python SDK with one command (Python 3.10–3.12):

```bash
pip install openspline
```

Load the generated service key and add the five-line client to your agent:

```bash
export OPENSPLINE_API_KEY=$(sed -n 's/^OPENSPLINE_API_KEY=//p' ~/.local/share/openspline/.env)
```

```python
from openspline import Openspline
client = Openspline()
async with client.avatar("portrait.jpg", quality="low") as avatar:
    print(avatar.viewer_url)
    await avatar.stream(agent.audio_stream(), sample_rate=24000)
```

`stream()` accepts generated PCM audio and returns after playback finishes. WAV and MP3 files are also supported.

## Connectors

These snippets plug into an existing async agent. Openspline owns output playback; disable your previous speaker output to avoid playing audio twice. Provider credentials stay on your backend. Each connector links to a complete runnable example.

### OpenAI Realtime

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
            await handle_event(event) # your existing handler
    finally:
        adapter.close()
```

[Run the example](examples/python/openai_realtime.py): `OPENAI_API_KEY=... python examples/python/openai_realtime.py portrait.jpg`.

### LiveKit

Install `pip install 'livekit-agents[openai]==1.8.3'`. Add the avatar before starting your existing `AgentSession`; playback uses your normal LiveKit frontend.

```python
from openspline import Openspline
from openspline.integrations.livekit import AvatarSession

avatar = AvatarSession("portrait.jpg", client=Openspline(quality="low"))
await avatar.start(agent_session, room=ctx.room)
await agent_session.start(room=ctx.room, agent=agent)
```

[Run the example](examples/python/livekit_agent.py): `PORTRAIT=portrait.jpg python examples/python/livekit_agent.py dev`. Set `OPENAI_API_KEY`, `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET`. The Compose service includes LiveKit support.

### Gemini Realtime

Install `pip install google-genai==1.75.0`. Forward every audio part from your existing Gemini Live session, including interruption and turn-end events.

```python
from openspline import Openspline
from openspline.integrations import GeminiLive

async with Openspline().avatar("portrait.jpg") as avatar:
    print(avatar.viewer_url)
    await avatar.wait_for_viewer()
    async for event in GeminiLive(avatar).wrap(live_session.receive()):
        await handle_event(event) # your existing handler
    await avatar.end_turn()
```

[Run the example](examples/python/gemini.py): `GEMINI_API_KEY=... python examples/python/gemini.py portrait.jpg`. Repeat the receive loop for subsequent turns; set `GEMINI_LIVE_MODEL` to override the model.

### Custom connector

Use any TTS provider or agent with an async PCM16 audio iterator. `stream()` buffers audio, flushes the final block, and waits for playback to finish.

```python
from openspline import Openspline
client = Openspline(quality="low")
async with client.avatar("portrait.jpg") as avatar:
    print(avatar.viewer_url)
    await avatar.stream(agent.audio_stream(), sample_rate=24000)
```

For callbacks, await `avatar.send_audio(chunk, sample_rate=24000)`, then `await avatar.end_turn()` when speech ends. Call `await avatar.interrupt()` on barge-in. Float32 uses `encoding="pcm_f32le"`; encoded streams use `encoding="mp3"`. [File example](examples/python/universal.py).

More connectors: [OpenAI Agents](examples/python/openai_agents.py), [Pipecat](examples/python/pipecat_bot.py), [Google ADK](examples/python/google_adk.py), [Vocode](examples/vocode/main.py), and [TEN](examples/ten/graph.json). Python examples have [locked dependencies](examples/python/requirements.lock); Vocode uses its [own environment](examples/vocode/requirements.lock).

## Configuration

Pass settings directly when initializing, or reuse a typed config object:

```python
from openspline import Openspline, OpensplineConfig

client = Openspline(url="http://localhost:7860", api_key="your-service-key", quality="low")
config = OpensplineConfig(quality="high", timeout=300, viewer_timeout=90)
client = Openspline(config=config) # a dict works too
```

| Python / Node option | Default | Purpose |
| --- | --- | --- |
| `url` | `OPENSPLINE_URL` or `http://localhost:7860` | Service address |
| `api_key` / `apiKey` | `OPENSPLINE_API_KEY` or empty | Backend credential |
| `quality` | `low` | Default quality: `low` or `high` |
| `timeout` | 180 s / 180,000 ms | HTTP and audio request timeout |
| `viewer_timeout` / `viewerTimeout` | 60 s / 60,000 ms | Wait for the playback destination |

Explicit Python parameters override the config object; explicit settings override environment defaults. `avatar(..., quality="high")` overrides that session only. Invalid configuration fails before connecting. [JSON Schema](config.schema.json) describes Python settings, with `#/$defs/node` and `#/$defs/server` for Node and worker configuration. Keep client configuration private; only `avatar.session` is browser-safe.

## Node

Use Node 20+ and install the SDK with one command:

```bash
npm install @openspline/node
```

```ts
import { Openspline, type OpensplineConfig } from "@openspline/node";
const config: OpensplineConfig = { quality: "low", timeout: 180000 };
const client = new Openspline(config);
await using avatar = await client.avatar("portrait.jpg");
console.log(avatar.viewerUrl);
await avatar.stream(agent.audioStream(), { sample_rate: 24000 });
```

[Node examples](examples/node): run `npm ci` there, then `node universal.mjs portrait.jpg speech.wav`. Without `await using`, call `avatar.close()` in `finally`.

## Frontend

Pass `avatar.session` from your backend to your frontend. Never expose the service API key or publisher token.

```tsx
import { Avatar } from "@openspline/react";

<Avatar session={session} />
```

For vanilla JS, Vue, or Svelte, call `registerAvatarElement()` from `@openspline/browser/element`, then set `document.querySelector("openspline-avatar").session = session`. Both players handle connection, autoplay permission, and cleanup. React also exports `useAvatar()`.

Install the component you need with `npm install @openspline/react` or `npm install @openspline/browser`.

## GPU workers

Edit [workers.yaml](workers.yaml). Each worker loads weights once and serves one session at a time; device assignments cannot overlap.

```yaml
workers:
  - { id: low-0, quality: low, devices: [0] }
  - { id: low-1, quality: low, devices: [1] }
```

For high quality, set a worker's quality to `high` and `OPENSPLINE_DOWNLOAD_QUALITY=all` before Compose startup. Busy workers return a typed `CapacityError`; an unconfigured quality returns `ConfigurationError`. A high-quality worker can reserve multiple GPUs with the server's `distributed` extra. IDs are relative to `CUDA_VISIBLE_DEVICES` when set.

For remote playback, set `OPENSPLINE_PUBLIC_URL`, use HTTPS and reachable WebRTC UDP ports, and configure `ice_servers` for TURN when needed. Compose uses Linux host networking. API reference: **http://localhost:7860/docs**. Health: `/readyz`; authenticated metrics: `/metrics`.

<details>
<summary>Development and validation</summary>

Run `uv sync --all-packages`, `npm ci && npm run build`, then `uv run pytest && npm test`. `uv run openspline serve --backend test --dev` starts a static-portrait development service without a GPU. Keep optional framework environments separate; from `examples/python`, use `uv pip sync requirements.lock` in an activated environment. Release workflows build and optionally publish packages and containers.

Real low/high inference, GPU streaming, browser/React playback, and native LiveKit have been exercised. Short A100 tests measured about 64 FPS for low quality and 8 FPS for high quality under different load conditions; these are engine rates, not playback latency. Long-duration, multi-GPU, cloud-provider, TEN runtime, and Docker deployment validation remain incomplete.

</details>

## License

[Apache-2.0](LICENSE). Vendored code retains its [third-party notices](packages/server/NOTICE); model weights have separate terms.

Built on [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead).
