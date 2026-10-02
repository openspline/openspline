import {RealtimeAgent,RealtimeSession} from '@openai/agents/realtime';
import {AvatarRealtimeTransport} from '@openspline/node/openai-agents';
import {Openspline} from '@openspline/node';
const avatar=await new Openspline().avatar(process.argv[2]);
try{
 console.log(avatar.viewerUrl);await avatar.waitForViewer();
 const transport=new AvatarRealtimeTransport(avatar);
 const session=new RealtimeSession(new RealtimeAgent({name:'Guide',instructions:'Keep responses brief.'}),{transport});
 let done;const finished=new Promise(resolve=>{done=resolve;});session.on('audio_stopped',()=>done());
 try{await session.connect({apiKey:process.env.OPENAI_API_KEY});session.sendMessage('Introduce yourself briefly.');await finished;await transport.drain();}finally{session.close();}
}finally{await avatar.close();}
