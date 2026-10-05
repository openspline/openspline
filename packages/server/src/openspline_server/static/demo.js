import{registerAvatarElement}from'./browser/element.js';registerAvatarElement();
import{waitForPlayback}from'./playback-ready.js';
import{prepareDemoQuality}from'./demo-quality.js';
import{setupGpuPicker}from'./demo-gpus.js';
const $=id=>document.getElementById(id);let session,ws,media,context,worklet,previewURL,playbackAbort,stopping,playbackError;
const gpuReady=fetch('/readyz').then(r=>r.json()).then(data=>{
 $('health').textContent=data.backend==='test'?'Test backend · static portrait':data.workers.some(w=>w.ready)?'GPU ready':'Worker unavailable';
 const qualities=data.demo_qualities??data.workers.map(w=>w.quality);
 for(const option of $('quality').options)option.disabled=!qualities.includes(option.value);
 if(data.workers.length===1)$('quality').value=data.workers[0].quality;
 if(!qualities.includes($('quality').value))$('quality').value=qualities[0]??'low';
}).catch(()=>{$('health').textContent='Service unavailable';}).then(()=>setupGpuPicker(request,$('quality'),$('gpu-settings'),$('gpu-list'),$('gpu-hint')));
$('portrait').onchange=()=>{const file=$('portrait').files[0];if(!file)return;if(previewURL)URL.revokeObjectURL(previewURL);previewURL=URL.createObjectURL(file);$('portrait-preview').src=previewURL;$('portrait-preview').hidden=false;$('empty').hidden=true;};
$('source').onchange=()=>{$('audio-label').hidden=$('source').value!=='file';};
async function request(path,init={}){const r=await fetch(path,init);if(!r.ok)throw new Error((await r.json()).error?.message??r.statusText);return r.status===204?null:r.json();}
let sequence=0;const pending=new Map();function send(type,body={}){const id=String(++sequence);return new Promise((resolve,reject)=>{pending.set(id,{resolve,reject});ws.send(JSON.stringify({type,id,...body}));});}
function connect(path,token){return new Promise((resolve,reject)=>{ws=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}${path}`);ws.onopen=()=>ws.send(JSON.stringify({token}));ws.onerror=()=>reject(new Error('Connection failed'));ws.onclose=()=>{const error=new Error('Connection closed');reject(error);for(const p of pending.values())p.reject(error);pending.clear();};ws.onmessage=({data})=>{const e=JSON.parse(data);if(e.type==='ready')resolve();if(e.type==='error'){reject(new Error(e.message));const p=pending.get(e.id);if(p){p.reject(new Error(e.message));pending.delete(e.id);}$('status').textContent=e.message;}else if(pending.has(e.id)){pending.get(e.id).resolve(e);pending.delete(e.id);}};});}
$('setup').onsubmit=async e=>{e.preventDefault();$('start').disabled=true;$('quality').disabled=true;$('gpu-settings').disabled=true;$('status').textContent='Preparing your avatar…';let gpuPicker;try{
 if(session)await stop(false);
 playbackError=null;
 const source=$('source').value,file=$('audio').files[0],quality=$('quality').value;
 gpuPicker=await gpuReady;gpuPicker.lock(true);const devices=gpuPicker.devices();
 if(source==='file'&&!file)throw new Error('Choose an audio file.');
 const form=new FormData();form.set('portrait',$('portrait').files[0]);form.set('quality',quality);
 await prepareDemoQuality(request,quality,message=>{$('status').textContent=message;},{devices});
 await gpuPicker.refresh().catch(()=>{});
 $('status').textContent='Preparing your avatar…';
 session=await request('/v1/sessions',{method:'POST',body:form});
 $('status').textContent='Connecting playback…';playbackAbort=new AbortController();
 const playbackReady=waitForPlayback($('avatar'),{signal:playbackAbort.signal,onBlocked:()=>{$('status').textContent='Click Enable playback on the avatar to continue.';}});playbackReady.catch(()=>{});
 const activeAvatar=$('avatar'),activeSession=session;
 activeAvatar.addEventListener('avatarerror',e=>{if(session!==activeSession||$('avatar')!==activeAvatar)return;playbackError=e.detail instanceof Error?e.detail:new Error(String(e.detail));$('status').textContent=playbackError.message;void stop(false);},{once:true});
 $('avatar').session={id:session.id,url:location.origin,token:session.token,transport:'websocket'};$('avatar').hidden=false;$('empty').hidden=true;$('portrait-preview').hidden=true;$('stop').hidden=false;
 await connect(`/v1/sessions/${session.id}/${source==='file'?'audio':'demo/'+source}`,session.publisher_token);
 await playbackReady;
 if(source==='file'){
   $('status').textContent='Generating synchronized audio and video…';const audio=new FormData();audio.set('audio',file);await request(`/v1/sessions/${session.id}/file`,{method:'POST',headers:{Authorization:`Bearer ${session.publisher_token}`},body:audio});await send('end_turn');$('status').textContent='Playback complete. Start another session to try a new voice.';
 }else{
   media=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true}});context=new AudioContext({sampleRate:24000});await context.audioWorklet.addModule('/static/microphone.js');worklet=new AudioWorkletNode(context,'pcm-capture');context.createMediaStreamSource(media).connect(worklet);worklet.port.onmessage=({data})=>{if(ws?.readyState===WebSocket.OPEN&&ws.bufferedAmount<100000)ws.send(data);};worklet.connect(context.destination);$('status').textContent='Connected. Speak naturally; interrupt at any time.';
 }
}catch(error){$('status').textContent=(playbackError??error).message;await stop(false);}finally{$('start').disabled=false;$('quality').disabled=false;gpuPicker?.lock(false);}};
async function stop(message=true){
 if(stopping)return stopping;
 stopping=(async()=>{
  playbackAbort?.abort();media?.getTracks().forEach(t=>t.stop());worklet?.disconnect();await context?.close();context=null;ws?.close();
  if(session){const old=session;session=null;await request(`/v1/sessions/${old.id}`,{method:'DELETE',headers:{Authorization:`Bearer ${old.publisher_token}`}}).catch(()=>{});}
  $('avatar')?.remove();const el=document.createElement('openspline-avatar');el.id='avatar';el.hidden=true;$('preview').append(el);$('portrait-preview').hidden=!previewURL;$('stop').hidden=true;if(message)$('status').textContent='Session ended.';
 })();try{await stopping;}finally{stopping=null;}
}
$('stop').onclick=()=>void stop();window.addEventListener('pagehide',()=>{media?.getTracks().forEach(t=>t.stop());ws?.close();});
