import test from 'node:test';
import assert from 'node:assert/strict';
import {ElevenLabsAgents} from '../packages/node/dist/integrations.js';

function setup(format) {
  const calls = [];
  const avatar = {
    sendAudio: async (raw, fmt) => calls.push(['audio', [...raw], fmt]),
    endTurn: async drain => calls.push(['end', drain]),
    interrupt: async () => calls.push(['interrupt']),
  };
  return {calls, adapter: new ElevenLabsAgents(avatar, format)};
}
const audio = (raw = [97, 98], event_id = 1, is_final = false) => ({
  type: 'audio', audio_event: {audio_base_64: Buffer.from(raw).toString('base64'), event_id, is_final},
});
const complete = event_id => ({type: 'agent_response_complete', agent_response_complete_event: {event_id}});

test('ElevenLabs negotiates PCM rates and flushes final chunks only once', async () => {
  for (const rate of [8000, 16000, 22050, 24000, 44100, 48000]) {
    const {calls, adapter} = setup();
    await adapter.handle({type: 'conversation_initiation_metadata', conversation_initiation_metadata_event: {agent_output_audio_format: `pcm_${rate}`}});
    await adapter.handle(audio());
    await adapter.handle({type: 'agent_response'});
    assert.deepEqual(calls, [['audio', [97, 98], {sample_rate: rate}]]);
    await adapter.handle(audio([99, 100], 1, true));
    await adapter.handle(complete(1));
    assert.deepEqual(calls.slice(1), [['audio', [99, 100], {sample_rate: rate}], ['end', false]]);
    await adapter.handle(audio([97, 98], 2));
    await adapter.handle(complete(2));
    assert.deepEqual(calls.at(-1), ['end', false]);
  }
});

test('ElevenLabs decodes mu-law and suppresses interrupted audio and completion', async () => {
  const {calls, adapter} = setup('ulaw_8000');
  await adapter.handle(audio([0, 128, 255, 127]));
  const expected = Buffer.alloc(8);
  [-32124, 32124, 0, 0].forEach((value, i) => expected.writeInt16LE(value, i * 2));
  assert.deepEqual(calls[0], ['audio', [...expected], {sample_rate: 8000}]);
  await adapter.handle({type: 'interruption', interruption_event: {event_id: 3}});
  await adapter.handle(audio([0], 3, true));
  await adapter.handle(complete(3));
  assert.equal(calls.length, 2);
  assert.deepEqual(calls.at(-1), ['interrupt']);
  await adapter.handle(audio([255], 4, true));
  assert.deepEqual(calls.at(-1), ['end', false]);
});

test('ElevenLabs preserves app events and rejects unsupported formats', async () => {
  const {calls, adapter} = setup();
  const events = ['client_tool_call', 'user_transcript', 'ping'].map(type => ({type}));
  async function* source() { yield* events; }
  const received = [];
  for await (const event of adapter.wrap(source())) received.push(event);
  assert.deepEqual(received, events);
  assert.deepEqual(calls, []);
  assert.throws(() => setup('mp3_44100_128'), /audio format/);
});
