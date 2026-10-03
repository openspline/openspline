import test from 'node:test';
import assert from 'node:assert/strict';
import {SocketPlayback} from '../packages/browser/dist/socket.js';

test('idle video advances without audio feedback and does not replace queued speech', async t => {
  const drawn=[],sent=[];
  const canvas={getContext:()=>({drawImage:image=>drawn.push(image)})};
  t.mock.method(globalThis,'createImageBitmap',async()=>({close(){}}));
  const player=new SocketPlayback({}, {muted:false});
  player.canvas=canvas;
  player.context={state:'running',currentTime:0};
  player.socket={readyState:1,send:raw=>sent.push(JSON.parse(raw))};
  const packet={type:'idle',epoch:0,sequence:1,image:btoa('image')};
  await player.receive(packet);
  assert.equal(drawn.length,1);
  assert.deepEqual(sent,[{type:'idle_played',sequence:1,epoch:0}]);
  assert.equal(player.end,0);
  player.frames=[{at:1,image:{close(){}}}];
  await player.receive({...packet,sequence:2});
  assert.equal(drawn.length,1);
  await player.receive({type:'interrupted',epoch:1});
  assert.equal(drawn.length,1); // Keep the last pose, never flash back to the portrait.
  await player.receive(packet);
  assert.equal(sent.length,2); // Stale idle frames cannot display or acknowledge.
  await player.receive({...packet,epoch:1});
  assert.equal(drawn.length,2);
  player.idle=false;
  player.portrait={};
  player.tick();
  assert.equal(drawn.length,2); // End-of-turn keeps the last video frame.
});

// Minimal DOM surfaces for the actual browser player; no network or audio devices.
globalThis.document={createElement:()=>({})};
globalThis.WebSocket={OPEN:1};
globalThis.createImageBitmap=async()=>({close(){}});
