import test from 'node:test';
import assert from 'node:assert/strict';
import {setupDemoCredentials} from '../packages/server/src/openspline_server/static/demo-credentials.js';

async function setup(request = async () => ({openai: {configured: true}, gemini: {configured: false}, elevenlabs: {configured: false, agent_configured: false}})) {
  const source = Object.assign(new EventTarget(), {value: 'file'});
  const input = Object.assign(new EventTarget(), {value: ''});
  const container = {}, label = {}, hint = {};
  const agent = {container: {}, input: {value: ''}, hint: {}};
  const credentials = await setupDemoCredentials(request, source, container, label, input, hint, agent);
  return {
    credentials, source, input, container, label, hint, agent,
    select(provider) { source.value = provider; source.dispatchEvent(new Event('change')); },
    type(key) { input.value = key; input.dispatchEvent(new Event('input')); },
  };
}

test('saved keys are indicated without putting a secret or placeholder in the submitted value', async () => {
  const ui = await setup();
  assert.equal(ui.container.hidden, true);
  assert.equal(ui.input.disabled, true);
  ui.select('openai');
  assert.equal(ui.container.hidden, false);
  assert.match(ui.input.placeholder, /saved key/);
  assert.equal(ui.input.value, '');
  assert.equal(ui.input.required, false);
  assert.equal(ui.credentials.key('openai'), undefined);
  ui.type('  entered-openai-key  ');
  assert.equal(ui.credentials.key('openai'), 'entered-openai-key');
  ui.type('');
  assert.equal(ui.credentials.key('openai'), undefined);
});

test('ElevenLabs requires an agent ID while allowing public agents without an API key', async () => {
  const ui = await setup();
  ui.select('elevenlabs');
  assert.equal(ui.input.required, false);
  assert.match(ui.label.textContent, /ElevenLabs.*optional/);
  assert.match(ui.hint.textContent, /Public agents/);
  assert.equal(ui.credentials.key('elevenlabs'), undefined);
  assert.equal(ui.agent.container.hidden, false);
  assert.equal(ui.agent.input.disabled, false);
  assert.equal(ui.agent.input.required, true);
  assert.throws(() => ui.credentials.agentId('elevenlabs'), /agent ID/);
  ui.agent.input.value = ' agent_test ';
  assert.equal(ui.credentials.agentId('elevenlabs'), 'agent_test');
  ui.type(' eleven-key ');
  ui.select('openai');
  assert.equal(ui.input.value, '');
  assert.equal(ui.agent.container.hidden, true);
  assert.equal(ui.agent.input.disabled, true);
  assert.equal(ui.agent.input.required, false);
  assert.equal(ui.credentials.agentId('openai'), undefined);
  ui.select('elevenlabs');
  assert.equal(ui.credentials.key('elevenlabs'), 'eleven-key');
  assert.equal(ui.credentials.agentId('elevenlabs'), 'agent_test');
  ui.agent.input.value = 'https://elevenlabs.io/agent';
  assert.throws(() => ui.credentials.agentId('elevenlabs'), /agent ID/);
});

test('ElevenLabs saved agent and API key are optional overrides, never placeholder values', async () => {
  const ui = await setup(async () => ({elevenlabs: {configured: true, agent_configured: true}}));
  ui.select('elevenlabs');
  assert.equal(ui.agent.input.required, false);
  assert.match(ui.agent.input.placeholder, /saved agent/);
  assert.equal(ui.credentials.agentId('elevenlabs'), undefined);
  assert.equal(ui.credentials.key('elevenlabs'), undefined);
  ui.agent.input.value = 'agent_override';
  assert.equal(ui.credentials.agentId('elevenlabs'), 'agent_override');
});

test('provider switching keeps keys separate and audio file mode needs no key', async () => {
  const ui = await setup();
  ui.select('openai');
  ui.type('openai-key');
  ui.select('gemini');
  assert.equal(ui.input.value, '');
  assert.equal(ui.label.textContent, 'Gemini API key');
  assert.equal(ui.input.required, true);
  ui.type('   ');
  assert.throws(() => ui.credentials.key('gemini'), /Enter your Gemini API key/);
  ui.type('gemini-key');
  ui.select('openai');
  assert.equal(ui.input.value, 'openai-key');
  assert.equal(ui.credentials.key('gemini'), 'gemini-key');
  ui.select('file');
  assert.equal(ui.container.hidden, true);
  assert.equal(ui.input.disabled, true);
  assert.equal(ui.input.required, false);
  assert.equal(ui.credentials.key('file'), undefined);
});

test('a failed status request still allows a session to use the server fallback', async () => {
  const ui = await setup(async () => { throw new Error('unavailable'); });
  ui.select('openai');
  assert.equal(ui.input.required, false);
  assert.equal(ui.credentials.key('openai'), undefined);
  assert.match(ui.hint.textContent, /leave blank/);
});
