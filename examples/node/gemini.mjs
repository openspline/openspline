import {GoogleGenAI,Modality} from '@google/genai';
import {Openspline} from '@openspline/node';
import {geminiLive,eventQueue} from '@openspline/node/integrations';
const avatar=await new Openspline().avatar(process.argv[2]);
try {
 console.log(avatar.viewerUrl);await avatar.waitForViewer();
 let done;const finished=new Promise(resolve=>{done=resolve;});
 const queue=eventQueue(async event=>{await geminiLive(avatar,event);if(event.serverContent?.turnComplete)done();});
 const session=await new GoogleGenAI({apiKey:process.env.GEMINI_API_KEY}).live.connect({model:process.env.GEMINI_LIVE_MODEL??'gemini-2.5-flash-native-audio-preview-12-2025',config:{responseModalities:[Modality.AUDIO]},callbacks:{onmessage:event=>queue.push(event),onerror:error=>console.error(error)}});
 try {session.sendClientContent({turns:[{role:'user',parts:[{text:'Introduce yourself briefly.'}]}],turnComplete:true});await finished;await queue.drain();await avatar.endTurn();}finally{session.close();}
}finally{await avatar.close();}
