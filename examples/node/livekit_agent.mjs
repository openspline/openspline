import {defineAgent,cli,WorkerOptions,voice} from '@livekit/agents';
import {realtime} from '@livekit/agents-plugin-openai';
import {LiveKitAvatarSession} from '@openspline/node/livekit';
import {fileURLToPath} from 'node:url';
export default defineAgent({entry:async ctx=>{
 await ctx.connect();
 const session=new voice.AgentSession({llm:new realtime.RealtimeModel()});
 const avatar=new LiveKitAvatarSession({portrait:process.env.PORTRAIT});
 await avatar.start(session,ctx.room);
 await session.start({room:ctx.room,agent:new voice.Agent({instructions:'Be helpful and concise.'})});
 await session.generateReply({instructions:'Greet the user.'});
}});
cli.runApp(new WorkerOptions({agent:fileURLToPath(import.meta.url)}));
