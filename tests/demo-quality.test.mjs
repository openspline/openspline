import test from 'node:test';
import assert from 'node:assert/strict';
import {prepareDemoQuality} from '../packages/server/src/openspline_server/static/demo-quality.js';

test('demo waits for model preparation and reports each stage',async()=>{
  const stages=['checking','unloading','loading','ready'],calls=[],messages=[];
  await prepareDemoQuality(async(path,init)=>{
    calls.push([path,init]);return {id:'job',state:stages.shift()};
  },'high',message=>messages.push(message),{devices:[0,1],pause:async()=>{}});
  assert.equal(JSON.parse(calls[0][1].body).quality,'high');
  assert.deepEqual(JSON.parse(calls[0][1].body).devices,[0,1]);
  assert.equal(calls[1][0],'/v1/demo/quality/job');
  assert.equal(calls.length,4);
  assert.match(messages[0],/Checking local high/);
  assert.match(messages[2],/Loading high/);
  assert.match(messages[2],/2 GPUs/);
});

test('preparation errors prevent session creation and explain recovery',async()=>{
  await assert.rejects(prepareDemoQuality(async()=>({state:'error',error:'GPU out of memory. Low quality is still available.'}),'high',()=>{}),/Low quality is still available/);
});
