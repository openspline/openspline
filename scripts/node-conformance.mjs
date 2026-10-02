import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {Openspline,CapacityError} from '../packages/node/dist/index.js';
const fixture=JSON.parse(await readFile('protocol/fixtures.json','utf8'));
const client=new Openspline();const avatar=await client.avatar(process.argv[2]);
try{
 assert.equal('publisher_token' in avatar.session,false);
 await assert.rejects(client.avatar(process.argv[2]),CapacityError);
 await avatar.sendAudio(Buffer.from(fixture.audio.data,'base64'),fixture.audio.format);
 assert.equal((await avatar.endTurn(false)).target_samples,4);
 await avatar.interrupt();assert.equal(avatar.epoch,1);
}finally{await avatar.close();}
const next=await client.avatar(process.argv[2]);await next.close();
console.log('Node shared protocol, typed capacity, interruption, and worker reuse passed');
