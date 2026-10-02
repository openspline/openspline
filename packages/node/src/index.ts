import { readFile } from 'node:fs/promises';
import WebSocket from 'ws';
import {resolveConfig,type OpensplineConfig,type Quality} from './config.js';
export type {OpensplineConfig,Quality} from './config.js';
export type AudioFormat = {sample_rate?:number;channels?:number;encoding?:'pcm_s16le'|'pcm_f32le'|'mp3'};
export interface SessionDescriptor {id:string;url:string;token:string;viewer_url:string;expires_at:number}
export class OpensplineError extends Error { constructor(message:string, public code:string) {super(message);} }
export class CapacityError extends OpensplineError {}
export class ConfigurationError extends OpensplineError {}
export class ConnectionError extends OpensplineError {}
export class InferenceError extends OpensplineError {}
function failure(code:string,message:string) { const C = ({capacity:CapacityError,configuration:ConfigurationError,connection:ConnectionError,inference:InferenceError} as Record<string,typeof OpensplineError>)[code] ?? OpensplineError;return new C(message,code); }
export class Openspline {
  readonly url:string;readonly apiKey:string;readonly timeout:number;
  readonly config:Readonly<Required<OpensplineConfig>>;
  constructor(options:OpensplineConfig={}) {this.config=resolveConfig(options);this.url=this.config.url;this.apiKey=this.config.apiKey;this.timeout=this.config.timeout;}
  async avatar(portrait:string|Uint8Array,options:{quality?:Quality}={}):Promise<AvatarSession> {
    const quality=options.quality??this.config.quality;if(!['low','high'].includes(quality))throw new ConfigurationError('quality must be low or high','configuration');
    const form=new FormData();const bytes=typeof portrait==='string'?await readFile(portrait):portrait;
    form.set('portrait',new Blob([new Uint8Array(bytes)]),'portrait');form.set('quality',quality);
    const body=await this.request('/v1/sessions',{method:'POST',headers:{Authorization:`Bearer ${this.apiKey}`},body:form});
    const avatar=new AvatarSession(this,body);
    try {await avatar.connect();return avatar;} catch(error) {await avatar.close();throw error;}
  }
  async request(path:string,init:RequestInit={}) {
    const response=await fetch(this.url+path,{...init,signal:AbortSignal.timeout(this.timeout)});
    if(!response.ok){const body=await response.json().catch(()=>({}));throw failure(body.error?.code??'request',body.error?.message??response.statusText);}
    return response.status===204?undefined:response.json();
  }
}
export class AvatarSession {
  readonly id:string;viewerUrl:string;session:SessionDescriptor;
  private token:string;private ws?:WebSocket;private pending=new Map<string,{resolve:(value:any)=>void;reject:(error:Error)=>void;timer:ReturnType<typeof setTimeout>}>();
  private callbacks=new Set<(event:any)=>void>();private heartbeat?:ReturnType<typeof setInterval>;private ready=false;private closed=false;private dirty=false;
  epoch=0;
  constructor(private client:Openspline,body:any){this.id=body.id;this.viewerUrl=body.viewer_url;this.token=body.publisher_token;const {publisher_token,...descriptor}=body;this.session=descriptor;}
  async connect(){
    const ws=this.ws=new WebSocket(this.client.url.replace(/^http/,'ws')+`/v1/sessions/${this.id}/audio`);
    ws.on('message',raw=>{const event=JSON.parse(raw.toString());this.epoch=Math.max(this.epoch,event.epoch??this.epoch);if(event.type==='viewer_ready'||event.viewer_ready)this.ready=true;if(event.type==='viewer_disconnected')this.ready=false;
      const pending=this.pending.get(event.id);
      if(pending){clearTimeout(pending.timer);this.pending.delete(event.id);event.type==='error'?pending.reject(failure(event.code,event.message)):pending.resolve(event);}
      else if(event.type==='error'&&!event.id)this.rejectAll(failure(event.code,event.message));
      for(const cb of this.callbacks)cb(event);
    });
    ws.on('close',()=>this.rejectAll(new ConnectionError('Session connection closed','connection')));
    await new Promise<void>((resolve,reject)=>{ws.once('open',()=>{ws.send(JSON.stringify({token:this.token}));resolve();});ws.once('error',reject);});
    this.heartbeat=setInterval(()=>void this.request('ping').catch(()=>{}),15000);this.heartbeat.unref();
  }
  private rejectAll(error:Error){for(const p of this.pending.values()){clearTimeout(p.timer);p.reject(error);}this.pending.clear();}
  on(callback:(event:any)=>void){this.callbacks.add(callback);return ()=>this.callbacks.delete(callback);}
  private request(type:string,data:Record<string,unknown>={}):Promise<any>{
    if(this.closed||this.ws?.readyState!==WebSocket.OPEN)return Promise.reject(new ConnectionError('Not connected','connection'));
    const id=crypto.randomUUID();
    return new Promise((resolve,reject)=>{const timer=setTimeout(()=>{this.pending.delete(id);reject(new ConnectionError('Request timed out','connection'));},this.client.timeout);this.pending.set(id,{resolve,reject,timer});this.ws!.send(JSON.stringify({type,id,...data}));});
  }
  async waitForViewer(timeout=this.client.config.viewerTimeout){if(this.ready)return;await new Promise<void>((resolve,reject)=>{const stop=this.on(event=>{if(event.type==='viewer_ready'||event.viewer_ready){clearTimeout(timer);stop();resolve();}});const timer=setTimeout(()=>{stop();reject(new ConnectionError('Playback destination did not connect','connection'));},timeout);});}
  async sendAudio(data:Uint8Array,format:AudioFormat={}){
    const fmt={sample_rate:24000,channels:1,encoding:'pcm_s16le',...format};if(![8000,16000,22050,24000,32000,44100,48000].includes(fmt.sample_rate)||![1,2].includes(fmt.channels)||!['pcm_s16le','pcm_f32le','mp3'].includes(fmt.encoding))throw new ConfigurationError('Invalid audio format','configuration');const size=fmt.encoding==='mp3'?16384:Math.floor(fmt.sample_rate*fmt.channels*(fmt.encoding==='pcm_f32le'?4:2)/10);
    const epoch=this.epoch;for(let i=0;i<data.length;i+=size){if(epoch!==this.epoch)return;await this.request('audio',{epoch,data:Buffer.from(data.subarray(i,i+size)).toString('base64'),format:fmt});this.dirty=true;}
  }
  async stream(source:AsyncIterable<Uint8Array>,format:AudioFormat={}){await this.waitForViewer();try{for await(const chunk of source)await this.sendAudio(chunk,format);await this.endTurn();}catch(error){await this.interrupt().catch(()=>{});throw error;}}
  async endTurn(drain=true){const result=await this.request('end_turn',{drain});this.dirty=false;return result;}
  async interrupt(){const body=await this.client.request(`/v1/sessions/${this.id}/interrupt`,{method:'POST',headers:{Authorization:`Bearer ${this.token}`}});this.epoch=body.epoch;this.dirty=false;}
  async playFile(path:string){await this.waitForViewer();const form=new FormData();form.set('audio',new Blob([new Uint8Array(await readFile(path))]),path.split('/').pop());await this.client.request(`/v1/sessions/${this.id}/file`,{method:'POST',headers:{Authorization:`Bearer ${this.token}`},body:form});this.dirty=true;await this.endTurn();}
  async viewer(){const descriptor=await this.client.request(`/v1/sessions/${this.id}/viewer`,{method:'POST',headers:{Authorization:`Bearer ${this.token}`}});this.session=descriptor;this.viewerUrl=descriptor.viewer_url;return descriptor as SessionDescriptor;}
  async attachLiveKit(options:{url:string;token:string;sender_identity:string}){return this.client.request(`/v1/sessions/${this.id}/livekit`,{method:'POST',headers:{Authorization:`Bearer ${this.token}`,'Content-Type':'application/json'},body:JSON.stringify(options)});}
  async close(){if(this.closed)return;this.closed=true;clearInterval(this.heartbeat);try{await this.client.request(`/v1/sessions/${this.id}`,{method:'DELETE',headers:{Authorization:`Bearer ${this.token}`}});}finally{this.ws?.close();this.rejectAll(new ConnectionError('Session closed','connection'));}}
  async [Symbol.asyncDispose](){try{if(this.dirty)await this.endTurn();}finally{await this.close();}}
}
