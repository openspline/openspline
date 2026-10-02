import test from 'node:test';
import assert from 'node:assert/strict';
import {nextPlayoutTime} from '../packages/browser/dist/playout.js';

test('a packet arriving just before its deadline stays contiguous',()=>{
  // The previous max(now + 40ms, end) inserted 20ms of silence here.
  assert.equal(nextPlayoutTime(1,1.02),1.02);
});

test('startup and true underruns rebuild a jitter buffer',()=>{
  assert.equal(nextPlayoutTime(1,0),1.16);
  assert.equal(nextPlayoutTime(1,0.99),1.16);
});

test('network jitter does not move already queued playback',()=>{
  let end=0;
  for(let i=0;i<50;i++){
    const arrival=i*0.04+(i%2 ? 0.05 : 0);
    const start=nextPlayoutTime(arrival,end);
    if(i)assert.equal(start,end);
    end=start+0.04;
  }
  assert.ok(Math.abs(end-2.16)<1e-9);
});
