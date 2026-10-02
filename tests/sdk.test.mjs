import test from 'node:test';import assert from 'node:assert/strict';
import {renderToString} from 'react-dom/server';import React from 'react';
import {Avatar} from '../packages/react/dist/index.js';
import {registerAvatarElement} from '../packages/browser/dist/element.js';
import {geminiLive,openaiRealtime} from '../packages/node/dist/integrations.js';
import {Openspline} from '../packages/node/dist/index.js';
test('browser and React imports are SSR safe',()=>{registerAvatarElement();assert.match(renderToString(React.createElement(Avatar,{session:{id:'s',url:'http://localhost',token:'x'}})),/Live avatar/);});
test('Gemini forwards all audio parts and turn end',async()=>{const calls=[];const avatar={sendAudio:async(...args)=>calls.push(args),endTurn:async()=>calls.push('end'),interrupt:async()=>calls.push('interrupt')};await geminiLive(avatar,{serverContent:{modelTurn:{parts:[{inlineData:{mimeType:'audio/pcm;rate=24000',data:'YWI='}},{inlineData:{mimeType:'audio/pcm;rate=16000',data:'Y2Q='}}]},turnComplete:true}});assert.equal(calls.length,3);assert.equal(calls[1][1].sample_rate,16000);});
test('SDK defaults to local service',()=>{assert.equal(new Openspline({url:'http://localhost:7860/'}).url,'http://localhost:7860');});
import {OpenAIRealtime,openaiAgents} from '../packages/node/dist/integrations.js';
test('OpenAI truncation uses played audio and Agents generation end flushes',async()=>{
 let callback;const sent=[],calls=[];const avatar={on:cb=>{callback=cb;return()=>{};},sendAudio:async()=>{},interrupt:async()=>calls.push('interrupt'),endTurn:async()=>calls.push('end')};
 const adapter=new OpenAIRealtime(avatar,e=>sent.push(e));
 await adapter.handle({type:'response.output_audio.delta',item_id:'item',content_index:0,delta:Buffer.alloc(48000).toString('base64')});
 callback({type:'playback',samples:12000});await adapter.handle({type:'input_audio_buffer.speech_started'});
 assert.equal(sent[0].audio_end_ms,250);await openaiAgents(avatar,{type:'audio_stopped'});assert.equal(calls.at(-1),'end');adapter.close();
});
import {AvatarSession} from '../packages/node/dist/index.js';
test('client config overrides environment and is immutable',()=>{
 const oldUrl=process.env.OPENSPLINE_URL,oldKey=process.env.OPENSPLINE_API_KEY;
 try{
  process.env.OPENSPLINE_URL='not a url';process.env.OPENSPLINE_API_KEY='environment-key';
  const options={url:'https://service.example///',apiKey:'',quality:'high',timeout:1000,viewerTimeout:1};
  const client=new Openspline(options);options.quality='low';
  assert.equal(client.url,'https://service.example');assert.equal(client.apiKey,'');assert.equal(client.config.quality,'high');assert.ok(Object.isFrozen(client.config));
 }finally{if(oldUrl===undefined)delete process.env.OPENSPLINE_URL;else process.env.OPENSPLINE_URL=oldUrl;if(oldKey===undefined)delete process.env.OPENSPLINE_API_KEY;else process.env.OPENSPLINE_API_KEY=oldKey;}
});
test('invalid configuration fails before network requests',()=>{
 for(const config of [{quality:'pro'},{timeout:0},{timeout:Infinity},{viewerTimeout:NaN},{viewerTimeout:.5},{url:'ftp://example.com'},{url:'https://user:secret@example.com'},{url:''},{apiKey:123},{quailty:'high'}])assert.throws(()=>new Openspline(config),TypeError);
});
test('configured quality reaches the session request and can be overridden',async()=>{
 const client=new Openspline({quality:'high'});const seen=[];
 client.request=async(path,init)=>{seen.push(init.body.get('quality'));throw new Error('captured before connect');};
 await assert.rejects(client.avatar(new Uint8Array()),/captured/);
 await assert.rejects(client.avatar(new Uint8Array(),{quality:'low'}),/captured/);
 assert.deepEqual(seen,['high','low']);
});
test('viewer timeout uses initialization config',async()=>{
 const avatar=new AvatarSession(new Openspline({viewerTimeout:1}),{id:'test',publisher_token:'private'});
 await assert.rejects(avatar.waitForViewer(),/Playback destination did not connect/);
});
