import type { AvatarSession } from './index.js';
/** Feed an event from the application's existing receive loop; never consume it twice. */
export async function openaiRealtime(avatar:AvatarSession,event:any){
  if(event.type==='response.output_audio.delta')await avatar.sendAudio(Buffer.from(event.delta,'base64'));
  else if(event.type==='response.output_audio.done')await avatar.endTurn(false);
  else if(event.type==='input_audio_buffer.speech_started')await avatar.interrupt();
}
export async function openaiAgents(avatar:AvatarSession,event:any){
  if(event.type==='audio')await avatar.sendAudio(new Uint8Array(event.data));
  else if(event.type==='audio_interrupted')await avatar.interrupt();
  else if(event.type==='audio_end'||event.type==='audio_stopped')await avatar.endTurn(false);
}
export async function geminiLive(avatar:AvatarSession,event:any){
  const content=event.serverContent;if(!content)return;
  if(content.interrupted){await avatar.interrupt();return;}
  for(const part of content.modelTurn?.parts??[]){const blob=part.inlineData;if(blob?.mimeType?.startsWith('audio/'))await avatar.sendAudio(Buffer.from(blob.data,'base64'),{sample_rate:Number(/rate=(\d+)/.exec(blob.mimeType)?.[1]??24000)});}
  if(content.turnComplete)await avatar.endTurn(false);
}
/** Queue event processing when a provider uses synchronous callbacks. Await drain() at shutdown. */
export function eventQueue(handle:(event:any)=>Promise<void>,limit=128){let queue=Promise.resolve();let pending=0;return {push(event:any){if(pending>=limit)throw new Error('Provider event queue is full; pause the producer');pending++;queue=queue.then(()=>handle(event)).finally(()=>{pending--;});return queue;},drain(){return queue;}};}

export class OpenAIRealtime {
  private item?:string;private index=0;private generated=0;private played=0;private base=0;private stop:()=>unknown;
  constructor(private avatar:AvatarSession,private send?:(event:unknown)=>unknown){this.stop=avatar.on(event=>{if(event.type==='playback')this.played=event.samples/48;});}
  async handle(event:any){
    if(event.type==='response.output_audio.delta'){
      if(event.item_id!==this.item){this.item=event.item_id;this.base=this.generated;}
      this.index=event.content_index??0;const raw=Buffer.from(event.delta,'base64');this.generated+=raw.length/48;await this.avatar.sendAudio(raw);
    }else if(event.type==='response.output_audio.done')await this.avatar.endTurn(false);
    else if(event.type==='input_audio_buffer.speech_started'){
      const heard=Math.max(0,this.played-this.base);await this.avatar.interrupt();
      if(this.send&&this.item)await this.send({type:'conversation.item.truncate',item_id:this.item,content_index:this.index,audio_end_ms:Math.floor(heard)});
      this.item=undefined;this.generated=this.played=this.base=0;
    }
  }
  close(){this.stop();}
}
