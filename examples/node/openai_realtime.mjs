import WebSocket from 'ws';
import {Openspline} from '@openspline/node';
import {OpenAIRealtime,eventQueue} from '@openspline/node/integrations';
const avatar=await new Openspline().avatar(process.argv[2]);
try{
 console.log(avatar.viewerUrl);await avatar.waitForViewer();
 const ws=new WebSocket(`wss://api.openai.com/v1/realtime?model=${process.env.OPENAI_REALTIME_MODEL??'gpt-realtime'}`,{headers:{Authorization:`Bearer ${process.env.OPENAI_API_KEY}`}});
 const adapter=new OpenAIRealtime(avatar,event=>ws.send(JSON.stringify(event)));
 let done;const finished=new Promise(resolve=>{done=resolve;});
 const queue=eventQueue(async event=>{await adapter.handle(event);if(event.type==='response.done')done();});
 ws.on('message',raw=>queue.push(JSON.parse(raw.toString())));
 await new Promise((resolve,reject)=>{ws.once('open',resolve);ws.once('error',reject);});
 ws.send(JSON.stringify({type:'response.create',response:{instructions:'Introduce yourself briefly.',output_modalities:['audio']}}));
 try{await finished;await queue.drain();await avatar.endTurn();}finally{adapter.close();ws.close();}
}finally{await avatar.close();}
