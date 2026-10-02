import {OpenAIRealtimeWebSocket} from '@openai/agents/realtime';
import type {AvatarSession} from './index.js';
/** WebSocket transport extension that truncates at acknowledged avatar playback. */
export class AvatarRealtimeTransport extends OpenAIRealtimeWebSocket {
  private played=0;private generated=0;private base=0;private item?:string;private unsubscribe:()=>unknown;
  private queue=Promise.resolve();private epoch=0;
  constructor(private avatar:AvatarSession,options:ConstructorParameters<typeof OpenAIRealtimeWebSocket>[0]={}){
    super(options);
    this.unsubscribe=avatar.on(event=>{if(event.type==='playback')this.played=event.samples/48;});
    this.on('audio',event=>{if(this.currentItemId!==this.item){this.item=this.currentItemId;this.base=this.generated;}this.generated+=event.data.byteLength/48;const epoch=this.epoch;this.queue=this.queue.then(async()=>{if(epoch===this.epoch)await avatar.sendAudio(new Uint8Array(event.data));});});
    this.on('audio_done',()=>{const epoch=this.epoch;this.queue=this.queue.then(async()=>{if(epoch===this.epoch)await avatar.endTurn(false);});});
  }
  override _interrupt(_elapsed:number,cancelOngoingResponse=true){
    this.epoch++;
    super._interrupt(Math.max(0,this.played-this.base),cancelOngoingResponse);
    void this.avatar.interrupt();this.item=undefined;this.played=this.generated=this.base=0;
  }
  async drain(){await this.queue;await this.avatar.endTurn();}
  override close(){this.unsubscribe();super.close();}
}
