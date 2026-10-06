// Play one response. Use the web demo for microphone conversations.
import WebSocket from 'ws';
import {Openspline} from '@openspline/node';
import {ElevenLabsAgents, eventQueue} from '@openspline/node/integrations';

const agentId = process.env.ELEVENLABS_AGENT_ID;
if (!agentId) throw new Error('Set ELEVENLABS_AGENT_ID.');
const query = new URLSearchParams({agent_id: agentId});
let url = `wss://api.elevenlabs.io/v1/convai/conversation?${query}`;
if (process.env.ELEVENLABS_API_KEY) {
  const response = await fetch(`https://api.elevenlabs.io/v1/convai/conversation/get-signed-url?${query}`, {
    headers: {'xi-api-key': process.env.ELEVENLABS_API_KEY},
    signal: AbortSignal.timeout(20000),
  });
  if (!response.ok) throw new Error(`ElevenLabs authentication failed (HTTP ${response.status}).`);
  url = (await response.json()).signed_url;
}

const avatar = await new Openspline().avatar(process.argv[2]);
let socket, timer;
try {
  console.log(avatar.viewerUrl);
  await avatar.waitForViewer();
  const adapter = new ElevenLabsAgents(avatar);
  let complete, fail, heardAudio = false;
  const finished = new Promise((resolve, reject) => { complete = resolve; fail = reject; });
  socket = new WebSocket(url);
  const send = event => socket.send(JSON.stringify(event));
  const queue = eventQueue(async event => {
    await adapter.handle(event);
    if (event.type === 'ping') send({type: 'pong', event_id: event.ping_event.event_id});
    else if (event.type === 'conversation_initiation_metadata') send({type: 'user_message', text: 'Introduce yourself briefly.'});
    else if (event.type === 'client_error') throw new Error('ElevenLabs rejected the conversation. Check your agent configuration.');
    else if (event.type === 'client_tool_call') throw new Error('Add your client tool handler before using this agent.');
    else if (event.type === 'audio') {
      heardAudio = true;
      if (event.audio_event.is_final) complete();
    } else if (event.type === 'agent_response_complete' && heardAudio) complete();
  });
  socket.on('open', () => send({type: 'conversation_initiation_client_data'}));
  socket.on('message', raw => {
    try { queue.push(JSON.parse(raw.toString())).catch(fail); } catch (error) { fail(error); }
  });
  socket.on('error', () => fail(new Error('ElevenLabs connection failed. Check agent access and credentials.')));
  socket.on('close', () => fail(new Error('ElevenLabs closed before completing a response.')));
  timer = setTimeout(() => fail(new Error('Timed out waiting for ElevenLabs audio.')), 60000);
  try { await finished; } finally { socket.close(); await queue.drain(); }
  await avatar.endTurn();
} finally {
  clearTimeout(timer);
  socket?.close();
  await avatar.close();
}
