const names = {openai: 'OpenAI', gemini: 'Gemini', elevenlabs: 'ElevenLabs'};

export async function setupDemoCredentials(request, source, container, label, input, hint, agent) {
  const keys = {};
  let providers = {}, current = source.value;

  function describe() {
    const saved = providers[current]?.configured;
    input.required = saved === false && current in names && current !== 'elevenlabs';
    input.placeholder = saved ? '•••••••• (saved key)' : `Paste your ${names[current]} API key`;
    hint.textContent = input.value.trim()
      ? 'Used when you start a session. This key is not saved.'
      : saved
        ? 'Using your saved key. Enter another key to override it for this session.'
        : current === 'elevenlabs'
          ? 'Public agents need no key. For private agents, enter an API key with access to the agent. This key is not saved.'
        : saved === false
          ? 'Enter your API key to try realtime conversation. This key is not saved.'
          : 'Enter your API key, or leave blank to use a key configured on the server.';
    if (agent) {
      const visible = current === 'elevenlabs';
      const savedAgent = providers.elevenlabs?.agent_configured;
      agent.container.hidden = !visible;
      agent.input.disabled = !visible;
      agent.input.required = visible && savedAgent === false;
      agent.input.placeholder = savedAgent ? 'Using saved agent ID' : 'agent_…';
      agent.hint.textContent = savedAgent
        ? 'Leave blank to use the saved agent, or enter another agent ID for this session.'
        : 'Copy the agent ID from your ElevenLabs dashboard, or set ELEVENLABS_AGENT_ID on the server.';
    }
  }

  function render() {
    current = source.value;
    const visible = current in names;
    container.hidden = !visible;
    input.disabled = !visible;
    label.textContent = `${names[current] ?? ''} API key${current === 'elevenlabs' ? ' (optional)' : ''}`;
    input.value = keys[current] ?? '';
    describe();
  }

  input.addEventListener('input', () => {
    keys[current] = input.value;
    describe();
  });
  source.addEventListener('change', render);
  render();
  try {
    providers = await request('/v1/demo/providers', {cache: 'no-store'});
  } catch {
    // A failed status request must not prevent use of a server fallback.
  }
  render();

  return {
    key(provider) {
      if (!(provider in names)) return undefined;
      const key = (keys[provider] ?? '').trim();
      if (!key && providers[provider]?.configured === false && provider !== 'elevenlabs') {
        throw new Error(`Enter your ${names[provider]} API key.`);
      }
      return key || undefined;
    },
    agentId(provider) {
      if (provider !== 'elevenlabs') return undefined;
      const id = agent?.input.value.trim();
      if ((!id && providers.elevenlabs?.agent_configured === false) || (id && !/^[A-Za-z0-9_-]{1,256}$/.test(id))) {
        throw new Error('Enter a valid ElevenLabs agent ID.');
      }
      return id || undefined;
    },
  };
}
