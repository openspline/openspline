# OpenSpline: complete agent reference

Use this file as the single source of product instructions when building an OpenSpline integration. It includes every page in the Mintlify navigation.

## Instructions for agents

1. Choose the fastest route that fits the user: dashboard embed, Cloud API, or local NVIDIA GPU.
2. Keep `OPENSPLINE_API_KEY` and provider credentials on the backend. Pass only `avatar.session` to browser players.
3. Replace placeholders such as `YOUR_API_KEY`, `portrait.jpg`, `agent.audio_stream()`, and `session` with the application’s real values.
4. Use the runnable examples linked below for provider-specific event handling; the short snippets show integration points.
5. For session lifecycle, close Node sessions in `finally` and use Python `async with`. Wait for a viewer before generating speech when required.
6. Check the installed package interfaces and example versions before changing code. The repository source is authoritative if it differs from this guide.

## Start with OpenSpline

Source: `index.mdx`

Turn a portrait and agent audio into synchronized video and speech.

1. [Try an avatar online](https://dashboard.openspline.dev/explore). No install or GPU needed.
2. [Create and embed your own avatar](/get-started/cloud-dashboard), or [stream audio with a Cloud API key](/get-started/cloud-api).
3. Have an NVIDIA GPU? [Run OpenSpline locally](/get-started/self-host).

After the first avatar, use the [Python SDK](/sdk/python), [Node SDK](/sdk/node), or an [agent connector](/connectors/overview).

## Create and embed in Cloud

Source: `get-started/cloud-dashboard.mdx`

Create an avatar at [openspline.dev](https://openspline.dev): upload an image, select a voice, and set a prompt. Copy its path from the dashboard.

```html
<script src="https://dashboard.openspline.dev/embed.js" defer></script>
<openspline-cloud
  cloud-url="https://dashboard.openspline.dev"
  path="nikola-tesla1">
</openspline-cloud>
```

Replace `nikola-tesla1` with your avatar path. [Try a public avatar](https://dashboard.openspline.dev/nikola-tesla1) first.

To supply your own agent audio, use the [Cloud API](/get-started/cloud-api).

## Stream audio on Cloud

Source: `get-started/cloud-api.mdx`

Create a key in the [Cloud dashboard](https://dashboard.openspline.dev/developers). Keep it on your backend.

```bash
pip install openspline
export OPENSPLINE_API_KEY="YOUR_API_KEY"
```

```python
from openspline import Openspline

async with Openspline().avatar("portrait.jpg") as avatar:
    print(avatar.viewer_url)
    await avatar.stream(agent.audio_stream(), sample_rate=24000)
```

Open the printed URL to watch playback. `agent.audio_stream()` is your async iterator of PCM16 chunks; `sample_rate` is in Hz.

Use a portrait saved in the dashboard:

```python
async with Openspline().avatar(avatar_id="YOUR_AVATAR_ID") as avatar:
    print(avatar.viewer_url)
    await avatar.stream(agent.audio_stream(), sample_rate=24000)
```

Pass `avatar.session` to your [frontend](/guides/frontend). For Node, use the [Node SDK](/sdk/node).

## Run on your GPU

Source: `get-started/self-host.mdx`

On Linux or WSL2 with an NVIDIA GPU:

```bash
curl -fsSL https://raw.githubusercontent.com/openspline/openspline/main/install.sh | sh
sh ~/.local/share/openspline/run.sh
```

Open [http://localhost:7860](http://localhost:7860) for the demo. Then use the same service from your agent:

```bash
pip install openspline
```

```python
from openspline import Openspline

async with Openspline().avatar("portrait.jpg", quality="low") as avatar:
    print(avatar.viewer_url)
    await avatar.stream(agent.audio_stream(), sample_rate=24000)
```

Open the printed viewer URL. The default service URL is `http://localhost:7860`.

PyTorch uses CUDA 12.8 for Blackwell and CUDA 12.6 for older NVIDIA GPUs. AMD and Apple GPUs are unsupported. See [GPU workers](/reference/gpu-workers) for multiple GPUs and High quality.

## Python SDK

Source: `sdk/python.mdx`

```bash
pip install openspline
```

For Cloud, set `OPENSPLINE_API_KEY`. For a local service, leave it unset or set `OPENSPLINE_URL`.

```python
from openspline import Openspline

async with Openspline().avatar("portrait.jpg", quality="low") as avatar:
    print(avatar.viewer_url)
    await avatar.stream(agent.audio_stream(), sample_rate=24000)
```

Play a WAV or MP3 file:

```python
async with Openspline().avatar("portrait.jpg") as avatar:
    print(avatar.viewer_url)
    await avatar.play_file("speech.wav")
```

For callback based audio:

```python
await avatar.send_audio(chunk, sample_rate=24000)
await avatar.end_turn()
await avatar.interrupt()  # when the user interrupts
```

`send_audio()` defaults to mono PCM16. Use `encoding="pcm_f32le"` for Float32 or `encoding="mp3"` for MP3 chunks. `stream()` ends the turn and waits for playback. Call `await avatar.wait_for_viewer()` if your agent should wait until someone opens the player.

[Runnable Python example](https://github.com/openspline/openspline/blob/main/examples/python/universal.py) · [Configuration](/reference/configuration) · [Backgrounds](/guides/backgrounds)

## Node SDK

Source: `sdk/node.mdx`

Use Node 20+.

```bash
npm install @openspline/node
```

For Cloud, set `OPENSPLINE_API_KEY`. For a local service, leave it unset or set `OPENSPLINE_URL`.

```ts
import { Openspline } from "@openspline/node";

const client = new Openspline();
const avatar = await client.avatar("portrait.jpg");
try {
  console.log(avatar.viewerUrl);
  await avatar.stream(agent.audioStream(), { sample_rate: 24000 });
} finally {
  await avatar.close();
}
```

Play a WAV or MP3 file:

```js
const avatar = await new Openspline().avatar("portrait.jpg");
try {
  console.log(avatar.viewerUrl);
  await avatar.playFile("speech.wav");
} finally {
  await avatar.close();
}
```

For a saved Cloud avatar, call `client.avatar({ avatarId: "YOUR_AVATAR_ID" })`. The Node equivalents of `send_audio`, `end_turn`, and `interrupt` are `sendAudio`, `endTurn`, and `interrupt`.

[Runnable Node example](https://github.com/openspline/openspline/blob/main/examples/node/universal.mjs) · [Configuration](/reference/configuration) · [Backgrounds](/guides/backgrounds)

## Show the avatar

Source: `guides/frontend.mdx`

Send `avatar.session` from your backend to the browser. Keep your Cloud API key and publisher token on the backend.

## React

```bash
npm install @openspline/react
```

```tsx
import { Avatar } from "@openspline/react";

<Avatar session={session} />
```

The package also exports `useAvatar()`.

## Browser custom element

```bash
npm install @openspline/browser
```

```js
import { registerAvatarElement } from "@openspline/browser/element";

registerAvatarElement();
document.querySelector("openspline-avatar").session = session;
```

```html
<openspline-avatar></openspline-avatar>
```

Both players handle connection and autoplay permission. For the server's WebSocket transport, pass `{ ...session, transport: "websocket" }`; otherwise they use WebRTC. The built in `avatar.viewer_url` is the fastest way to watch a session.

## Photo and video backgrounds

Source: `guides/backgrounds.mdx`

Use a PNG portrait with its background removed. The portrait's alpha matte defines the silhouette; an opaque portrait keeps its original background. Images stay still and videos loop silently.

```python
async with Openspline().avatar("portrait.png", background="scene.mp4") as avatar:
    await avatar.play_file("speech.wav")
```

```js
const avatar = await new Openspline().avatar("portrait.png", {
  background: "scene.jpg",
});
```

Python accepts a file path or bytes; Node accepts a file path or `Uint8Array`. The HTTP API accepts a `background` file in `POST /v1/sessions`. Limits: 25 MiB, 16 megapixels, and video up to 60 seconds at 120 fps. Backgrounds are fixed for each session.

LiveKit, Pipecat, and TEN accept `background` on their avatar session or processor. Other adapters use the background already set on their shared session. Viewers receive composited video automatically.

## Choose a connector

Source: `connectors/overview.mdx`
OpenSpline replaces your agent's speaker output with an avatar. Keep provider credentials on your backend, and disable the previous speaker output to avoid double playback.

| Your agent | Start here |
| --- | --- |
| Any TTS or PCM audio stream | [Python SDK](/sdk/python) or [Node SDK](/sdk/node) |
| OpenAI Realtime API | [OpenAI Realtime](/connectors/openai-realtime) |
| OpenAI Agents SDK | [OpenAI Agents](/connectors/openai-agents) |
| LiveKit Agents (Python and Node) | [LiveKit plugins and demos](/connectors/livekit) |
| Gemini Live | [Gemini Live](/connectors/gemini) |
| ElevenLabs Agents | [ElevenLabs Agents](/connectors/elevenlabs) |
| Pipecat, Google ADK, Vocode, TEN | [Other integrations](/connectors/other) |

## OpenAI Realtime

Source: `connectors/openai-realtime.mdx`

```bash
pip install 'openai==2.54.0' openspline
```

Wrap your existing Realtime connection:

```python
from openspline import Openspline
from openspline.integrations import OpenAIRealtime

async with Openspline().avatar("portrait.jpg") as avatar:
    print(avatar.viewer_url)
    await avatar.wait_for_viewer()
    adapter = OpenAIRealtime(avatar, connection)
    try:
        async for event in adapter.wrap(connection):
            await handle_event(event)
    finally:
        adapter.close()
```

The adapter forwards tools and transcripts to your existing handler.

```bash
OPENAI_API_KEY=... python examples/python/openai_realtime.py portrait.jpg
```

[Python example](https://github.com/openspline/openspline/blob/main/examples/python/openai_realtime.py) · [Node example](https://github.com/openspline/openspline/blob/main/examples/node/openai_realtime.mjs)

## OpenAI Agents

Source: `connectors/openai-agents.mdx`

```bash
pip install 'openai-agents==0.20.0' openspline
```

```python
from agents.realtime import RealtimeAgent, RealtimeRunner
from openspline import Openspline
from openspline.integrations import OpenAIAgents

async with Openspline().avatar("portrait.jpg") as avatar:
    print(avatar.viewer_url)
    await avatar.wait_for_viewer()
    adapter = OpenAIAgents(avatar)
    runner = RealtimeRunner(RealtimeAgent(name="Guide", instructions="Keep responses brief."))
    async with await runner.run(model_config={"playback_tracker": adapter.tracker}) as session:
        await session.send_message("Introduce yourself in two sentences.")
        try:
            async for event in adapter.wrap(session):
                if event.type == "audio_end":
                    break
        finally:
            adapter.close()
        await avatar.end_turn()
```

[Python example](https://github.com/openspline/openspline/blob/main/examples/python/openai_agents.py) · [Node example](https://github.com/openspline/openspline/blob/main/examples/node/openai_agents.mjs)

## LiveKit Agents

Source: `connectors/livekit.mdx`

<img src="/images/openspline.png" alt="OpenSpline" width="64" height="64" />

Add an avatar to your existing LiveKit agent in three lines. OpenSpline publishes synchronized video and speech into your LiveKit room. Keep your models, voice, tools, and frontend.

<Tabs>
<Tab title="Python">
```python
from livekit.plugins import openspline

avatar = openspline.AvatarSession(avatar_id="YOUR_AVATAR_ID")
await avatar.start(session, room=ctx.room)
```
</Tab>
<Tab title="Node.js">
```typescript
import * as openspline from '@livekit/agents-plugin-openspline';

const avatar = new openspline.AvatarSession({ avatarId: 'YOUR_AVATAR_ID' });
await avatar.start(session, ctx.room);
```
</Tab>
</Tabs>

Start the avatar **before** `session.start()`. `avatar.start()` waits for the worker to publish video, so the first words have a face. Your voice provider still determines what the agent says and how it sounds; a saved avatar supplies its portrait and background, not its conversation or TTS configuration.

## Demos

Recorded from the Python plugin’s room tracks with LiveKit Egress, using High quality and each avatar's assigned voice synthesized with ElevenLabs Eleven v4 (`eleven_v4`). These are the actual LiveKit worker tracks. Enable sound to hear the speech; caption files are linked beneath each video.

{/* Each demo is self-contained so it can be removed independently. */}
### Professor Nikolaev

Jay Wayne — Wise University Professor

<video controls playsInline preload="metadata" poster="/videos/livekit/professor-nikolaev.jpg" style={{ width: "100%", maxWidth: "512px", borderRadius: "12px" }}>
  <source src="/videos/livekit/professor-nikolaev.mp4" type="video/mp4" />
</video>


### News-anchor

Kanika — Confident News Anchor

<video controls playsInline preload="metadata" poster="/videos/livekit/news-anchor.jpg" style={{ width: "100%", maxWidth: "512px", borderRadius: "12px" }}>
  <source src="/videos/livekit/news-anchor.mp4" type="video/mp4" />
</video>


### Meera

Zara — Soft, Professional and Gentle

<video controls playsInline preload="metadata" poster="/videos/livekit/meera.jpg" style={{ width: "100%", maxWidth: "512px", borderRadius: "12px" }}>
  <source src="/videos/livekit/meera.mp4" type="video/mp4" />
</video>




## Installation

Install the plugin in your LiveKit agent project:

<Tabs>
<Tab title="Python">
```bash
uv add "livekit-plugins-openspline~=0.1.0"
```
Requires Python 3.10–3.12.
</Tab>
<Tab title="Node.js">
```bash
pnpm add @livekit/agents-plugin-openspline
```
Requires Node.js 20 or later.
</Tab>
</Tabs>

## Configure and run

Create an OpenSpline key and copy an owned avatar's ID from [Developers](https://dashboard.openspline.dev/developers). An avatar's public URL path is different from its ID. Add these backend-only settings to your agent project's `.env` file:

```bash
OPENSPLINE_API_KEY=your-openspline-key
OPENSPLINE_AVATAR_ID=your-avatar-id
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your-livekit-key
LIVEKIT_API_SECRET=your-livekit-secret
OPENAI_API_KEY=your-openai-key
```

For a new agent, follow the [LiveKit Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai/), then add the three-line avatar setup above before `session.start()`. The `OPENAI_API_KEY` setting is needed only when using OpenAI.

See the complete OpenAI Realtime agent examples: [Python](https://github.com/livekit/agents/blob/main/examples/avatar/openspline_avatar.py) · [Node.js](https://github.com/livekit/agents-js/blob/main/examples/src/openspline_avatar.ts).

Connect using the [LiveKit Agent Console](https://docs.livekit.io/agents/start/console/) or a [frontend starter app](https://docs.livekit.io/frontends/start/). In React, `useVoiceAssistant()` associates the avatar's audio/video tracks with your agent. No OpenSpline viewer or second audio player is needed.

## Use a portrait or your own GPU

A portrait works on Cloud with your key, or against a [self-hosted service](/get-started/self-host) with the key unset:

<Tabs>
<Tab title="Python">
```python
avatar = openspline.AvatarSession(portrait="portrait.jpg", quality="high")
await avatar.start(session, room=ctx.room)
```
</Tab>
<Tab title="Node.js">
```typescript
const avatar = new openspline.AvatarSession({ portrait: 'portrait.jpg', quality: 'high' });
await avatar.start(session, ctx.room);
```
</Tab>
</Tabs>

Unset `OPENSPLINE_API_KEY` for self-hosting. `OPENSPLINE_URL` defaults to `http://localhost:7860` without a key and to the Cloud API with a key. For the runnable examples, unset `OPENSPLINE_AVATAR_ID` and set `PORTRAIT` to your file path. Cloud requires a publicly reachable `wss://` LiveKit server.

## Options and lifecycle

| Python / Node option | Purpose |
| --- | --- |
| `avatar_id` / `avatarId` | Owned saved Cloud avatar ID; requires a Cloud key |
| `portrait` | Image file path or bytes; supply this or an avatar ID |
| `quality` | `low` by default; `high` for High quality |
| `background` | Optional image/video path or bytes; overrides the saved background |
| `api_key` / `apiKey`, `url` | OpenSpline credentials and service address; defaults to `OPENSPLINE_*` |
| `client` | Existing OpenSpline client; use instead of `api_key`/`apiKey` and `url` |
| `livekit_url` / `livekitUrl` | LiveKit server address; defaults to `LIVEKIT_URL` |
| `livekit_api_key` / `livekitApiKey` | Defaults to `LIVEKIT_API_KEY` |
| `livekit_api_secret` / `livekitApiSecret` | Defaults to `LIVEKIT_API_SECRET` |
| `join_timeout` / `joinTimeout` | Video publication timeout after preparation: 30 seconds / 30000 milliseconds |

LiveKit closes the avatar at job shutdown. Outside a job, call `await avatar.aclose()` in `finally`. Create a new `AvatarSession` for each call. For longer GPU cold starts, inject an OpenSpline client with a larger `timeout` (seconds in Python, milliseconds in Node).

The worker is an agent participant with `lk.publish_on_behalf` pointing to your agent. Interruption and playback acknowledgements travel over LiveKit; the avatar reports standard join and playback metrics. Only the avatar plays the speech, avoiding duplicate room audio.

Existing `openspline.integrations.livekit` and `@openspline/node/livekit` imports remain supported. Their legacy `url`, `api_key`/`apiKey`, and `api_secret`/`apiSecret` arguments refer to **LiveKit**; configure OpenSpline through their `client` argument.

## Gemini Live

Source: `connectors/gemini.mdx`

```bash
pip install 'google-genai==1.75.0' openspline
```

```python
from openspline import Openspline
from openspline.integrations import GeminiLive

async with Openspline().avatar("portrait.jpg") as avatar:
    print(avatar.viewer_url)
    await avatar.wait_for_viewer()
    async for event in GeminiLive(avatar).wrap(live_session.receive()):
        await handle_event(event)
    await avatar.end_turn()
```

Repeat the receive loop for later turns. Set `GEMINI_API_KEY`; optionally set `GEMINI_LIVE_MODEL`.

```bash
GEMINI_API_KEY=... python examples/python/gemini.py portrait.jpg
```

[Python example](https://github.com/openspline/openspline/blob/main/examples/python/gemini.py) · [Node example](https://github.com/openspline/openspline/blob/main/examples/node/gemini.mjs)

## ElevenLabs Agents

Source: `connectors/elevenlabs.mdx`

Pass each decoded WebSocket event, including the initial metadata, to the adapter. Keep your input, pong, and tool handlers.

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
        await handle_event(event)
```

Set `ELEVENLABS_AGENT_ID`. Private agents also need `ELEVENLABS_API_KEY`; the runnable examples request a signed URL on the backend.

```bash
ELEVENLABS_AGENT_ID=... python examples/python/elevenlabs_agents.py portrait.jpg
```

The adapter supports PCM and μ-law, final audio flushes, and interruptions. If attaching after connection setup, pass the negotiated `audio_format`, such as `"pcm_24000"`. The demo plays one response; microphone conversations require your input handler.

[Python example](https://github.com/openspline/openspline/blob/main/examples/python/elevenlabs_agents.py) · [Node example](https://github.com/openspline/openspline/blob/main/examples/node/elevenlabs_agents.mjs)

## Other integrations

Source: `connectors/other.mdx`

Use the [Python SDK](/sdk/python) or [Node SDK](/sdk/node) for any async PCM16 audio stream.

```python
async with Openspline().avatar("portrait.jpg") as avatar:
    print(avatar.viewer_url)
    await avatar.stream(agent.audio_stream(), sample_rate=24000)
```

For callbacks, call `send_audio()` for each chunk, then `end_turn()` when speech ends and `interrupt()` on barge in.

| Integration | Runnable example |
| --- | --- |
| Pipecat | [Python bot](https://github.com/openspline/openspline/blob/main/examples/python/pipecat_bot.py) |
| Google ADK | [Python agent](https://github.com/openspline/openspline/blob/main/examples/python/google_adk.py) |
| Vocode | [Vocode app](https://github.com/openspline/openspline/blob/main/examples/vocode/main.py) |
| TEN | [TEN graph](https://github.com/openspline/openspline/blob/main/examples/ten/graph.json) |
| Any audio source | [Python universal](https://github.com/openspline/openspline/blob/main/examples/python/universal.py) · [Node universal](https://github.com/openspline/openspline/blob/main/examples/node/universal.mjs) |

The Python examples have [locked dependencies](https://github.com/openspline/openspline/blob/main/examples/python/requirements.lock). Vocode uses [its own lockfile](https://github.com/openspline/openspline/blob/main/examples/vocode/requirements.lock).

## Configuration

Source: `reference/configuration.mdx`

```python
from openspline import Openspline, OpensplineConfig

client = Openspline(url="http://localhost:7860", quality="low")
client = Openspline(config=OpensplineConfig(quality="high", timeout=300))
```

```ts
import { Openspline, type OpensplineConfig } from "@openspline/node";

const config: OpensplineConfig = { apiKey: "YOUR_API_KEY", quality: "low" };
const client = new Openspline(config);
```

| Python / Node option | Default | Use |
| --- | --- | --- |
| `api_key` / `apiKey` | `OPENSPLINE_API_KEY` | Cloud key, backend only |
| `url` | Cloud with a key; otherwise `http://localhost:7860` | Service address; `OPENSPLINE_URL` also works |
| `quality` | `low` | `low` or `high` |
| `timeout` | 180 s / 180,000 ms | Requests and audio |
| `viewer_timeout` / `viewerTimeout` | 60 s / 60,000 ms | Waiting for a viewer |

Explicit settings override environment defaults. Per avatar `quality` overrides the client setting. Only the session descriptor belongs in the browser.

[Full JSON Schema](https://github.com/openspline/openspline/blob/main/config.schema.json)

## GPU workers

Source: `reference/gpu-workers.mdx`

Edit `~/.local/share/openspline/workers.yaml`. Each worker runs one active session; GPU assignments cannot overlap.

```yaml
workers:
  - { id: low-0, quality: low, devices: [0] }
  - { id: low-1, quality: low, devices: [1] }
```

For High quality, assign a GPU pair and prepare the models:

```yaml
workers:
  - { id: high-0, quality: high, devices: [0, 1] }
```

```bash
sh ~/.local/share/openspline/run.sh --prepare
```

Remove `OPENSPLINE_GPU`, `OPENSPLINE_GPUS`, and `OPENSPLINE_QUALITY` overrides when using multiple workers. Set `OPENSPLINE_HOST`, `OPENSPLINE_PORT`, and `OPENSPLINE_PUBLIC_URL` in the installed `.env` for remote access. Use HTTPS for remote microphone access; WebRTC also needs reachable UDP ports or TURN in `ice_servers`.

Busy workers return `CapacityError`. An unconfigured quality returns `ConfigurationError`. Check `/readyz` and `/metrics` on the service.

[Example workers.yaml](https://github.com/openspline/openspline/blob/main/workers.yaml)

## Service HTTP API

Source: `reference/http-api.mdx`

The SDKs handle session creation, streaming, interruption, and cleanup. For direct HTTP use, open the running service's [interactive API reference](http://localhost:7860/docs).

| Endpoint | Use |
| --- | --- |
| `POST /v1/sessions` | Create a session with `portrait` or Cloud `avatar_id`, optional `background`, and `quality` |
| `GET /v1/sessions/{id}` | Read session state |
| `POST /v1/sessions/{id}/viewer` | Get a viewer descriptor |
| `POST /v1/sessions/{id}/file` | Play an audio file |
| `POST /v1/sessions/{id}/interrupt` | Stop current speech |
| `DELETE /v1/sessions/{id}` | Close a session |
| `GET /healthz`, `GET /readyz`, `GET /metrics` | Check health, readiness, and metrics |

Create a local session:

```bash
curl -X POST http://localhost:7860/v1/sessions \
  -F portrait=@portrait.jpg \
  -F quality=low
```

Audio streaming uses `WS /v1/sessions/{id}/audio`; browser playback uses WebRTC or `WS /v1/sessions/{id}/playback`. Use an SDK unless you need to implement the wire protocol.

## Develop and validate

Source: `reference/development.mdx`

From the repository root:

```bash
uv sync --all-packages
npm ci
npm run build
uv run pytest
npm test
```

Run a static portrait service without a GPU:

```bash
uv run openspline serve --backend test --dev
```

For Python integration examples, use the [example lockfile](https://github.com/openspline/openspline/blob/main/examples/python/requirements.lock) in a separate environment. Installer dependency locks can be regenerated with `python scripts/lock-gpu.py`.

The service uses one worker per active session. Local GPU, playback, and native LiveKit paths have been exercised; long duration, multi GPU, Cloud provider, and TEN runtime validation remain incomplete.
