export const examples = {
  elevenlabs: {
    title: 'ElevenLabs Agents',
    description: 'Connect your existing ElevenLabs agent WebSocket.',
    file: 'elevenlabs_agents.py',
    code: `import json
from openspline import Openspline
from openspline.integrations import ElevenLabsAgents

async def connect_avatar(connection):
    async with Openspline().avatar("portrait.jpg") as avatar:
        print(avatar.viewer_url)
        await avatar.wait_for_viewer()
        adapter = ElevenLabsAgents(avatar)
        async for message in connection:
            event = json.loads(message)
            await adapter.handle(event)  # Include initiation metadata.
            if event["type"] == "ping":
                await connection.send(json.dumps({
                    "type": "pong",
                    "event_id": event["ping_event"]["event_id"],
                }))
            # Keep your transcripts and client tool handlers here.`,
  },
  file: {
    title: 'Custom audio',
    description: 'Play an audio file, or stream PCM16 from your own agent.',
    file: 'universal.py',
    code: `from openspline import Openspline

async def play_audio(path):
    async with Openspline().avatar("portrait.jpg") as avatar:
        print(avatar.viewer_url)
        await avatar.play_file(path)  # WAV or MP3

# Custom connector: pass an async iterator of PCM16 chunks.
async def stream_audio(audio_chunks):
    async with Openspline().avatar("portrait.jpg") as avatar:
        print(avatar.viewer_url)
        await avatar.stream(audio_chunks, sample_rate=24000)`,
  },
  openai: {
    title: 'OpenAI Realtime',
    description: 'Connect your existing OpenAI Realtime session.',
    file: 'openai_realtime.py',
    code: `from openspline import Openspline
from openspline.integrations import OpenAIRealtime

async def connect_avatar(connection):
    async with Openspline().avatar("portrait.jpg") as avatar:
        print(avatar.viewer_url)
        await avatar.wait_for_viewer()
        adapter = OpenAIRealtime(avatar, connection)
        try:
            async for event in adapter.wrap(connection):
                pass  # Keep your existing event handler here.
        finally:
            adapter.close()`,
  },
  gemini: {
    title: 'Gemini Live',
    description: 'Connect your existing Gemini Live session.',
    file: 'gemini.py',
    code: `from openspline import Openspline
from openspline.integrations import GeminiLive

async def connect_avatar(live_session):
    async with Openspline().avatar("portrait.jpg") as avatar:
        print(avatar.viewer_url)
        await avatar.wait_for_viewer()
        adapter = GeminiLive(avatar)
        while True:
            async for event in adapter.wrap(live_session.receive()):
                pass  # Keep your existing event handler here.
            await avatar.end_turn()`,
  },
};

export function setupDemoExamples(source, title, description, code, link, copy, status) {
  function render() {
    const example = examples[source.value] ?? examples.file;
    title.textContent = example.title;
    description.textContent = example.description;
    code.textContent = example.code;
    link.href = `https://github.com/openspline/openspline/blob/main/examples/python/${example.file}`;
    status.textContent = '';
  }
  source.addEventListener('change', render);
  copy.addEventListener('click', async () => {
    const text = code.textContent;
    try {
      await navigator.clipboard.writeText(text);
      if (code.textContent === text) status.textContent = 'Code copied.';
    } catch {
      if (code.textContent !== text) return;
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      code.parentElement.focus();
      status.textContent = 'Code selected. Press Ctrl+C or Cmd+C to copy.';
    }
  });
  render();
}
