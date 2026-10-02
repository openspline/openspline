import {voice} from '@livekit/agents';
import type {Room} from '@livekit/rtc-node';
import {AccessToken} from 'livekit-server-sdk';
import {Openspline,type AvatarSession,type Quality} from './index.js';
export class LiveKitAvatarSession extends voice.AvatarSession{
  private avatar?:AvatarSession;private identity='openspline-'+crypto.randomUUID();
  constructor(private options:{portrait:string|Uint8Array;quality?:Quality;client?:Openspline;url?:string;apiKey?:string;apiSecret?:string}){super();}
  override get avatarIdentity(){return this.identity;}
  override get provider(){return 'openspline';}
  override async start(agentSession:voice.AgentSession,room:Room){
    await super.start(agentSession,room);
    if(!room.localParticipant||!room.name)throw new Error('Connect the agent to its room before starting the avatar');
    this.avatar=await (this.options.client??new Openspline()).avatar(this.options.portrait,{quality:this.options.quality});
    const token=new AccessToken(this.options.apiKey??process.env.LIVEKIT_API_KEY,this.options.apiSecret??process.env.LIVEKIT_API_SECRET,{identity:this.identity,attributes:{'lk.publish_on_behalf':room.localParticipant.identity}});
    token.kind='agent';token.addGrant({roomJoin:true,room:room.name});
    try{
      await this.avatar.attachLiveKit({url:this.options.url??process.env.LIVEKIT_URL!,token:await token.toJwt(),sender_identity:room.localParticipant.identity});
      agentSession.output.audio=new voice.DataStreamAudioOutput({room,destinationIdentity:this.identity,waitPlaybackStart:true});
    }catch(error){await this.aclose();throw error;}
  }
  override async aclose(){await this.avatar?.close();await super.aclose();}
}
