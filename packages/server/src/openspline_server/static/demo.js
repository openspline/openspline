import{registerAvatarElement}from'./browser/element.js';registerAvatarElement();
const $=id=>document.getElementById(id);let session,ws,media,context,worklet,reader,previewURL;
fetch('/readyz').then(r=>r.json()).then(data=>{$('health').textContent=data.backend==='test'?'Test backend · static portrait':data.workers.some(w=>w.ready)?'GPU ready':'Worker unavailable';}).catch(()=>{$('health').textContent='Service unavailable';});
$('portrait').onchange=()=>{const file=$('portrait').files[0];if(!file)return;if(previewURL)URL.revokeObjectURL(previewURL);previewURL=URL.createObjectURL(file);$('portrait-preview').src=previewURL;$('portrait-preview').hidden=false;$('empty').hidden=true;};
$('source').onchange=()=>{$('audio-label').hidden=$('source').value!=='file';};
async function request(path,init={}){const r=await fetch(path,init);if(!r.ok)throw new Error((await r.json()).error?.message??r.statusText);return r.status===204?null:r.json();}
const pending=new Map();function send(type,body={}){const id=crypto.randomUUID();return new Promise((resolve,reject)=>{pending.set(id,{resolve,reject});ws.send(JSON.stringify({type,id,...body}));});}
function connect(path,token){return new Promise((resolve,reject)=>{ws=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}${path}`);ws.onopen=()=>ws.send(JSON.stringify({token}));ws.onerror=()=>reject(new Error('Connection failed'));ws.onclose=()=>{for(const p of pending.values())p.reject(new Error('Connection closed'));pending.clear();};ws.onmessage=({data})=>{const e=JSON.parse(data);if(e.type==='ready')resolve();if(e.type==='error'){const p=pending.get(e.id);if(p){p.reject(new Error(e.message));pending.delete(e.id);}$('status').textContent=e.message;}else if(pending.has(e.id)){pending.get(e.id).resolve(e);pending.delete(e.id);}};});}
$('setup').onsubmit=async e=>{e.preventDefault();$('start').disabled=true;$('status').textContent='Preparing your avatar…';try{
 const form=new FormData();form.set('portrait',$('portrait').files[0]);form.set('quality',$('quality').value);
 const playbackReady=new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(new Error('Playback did not connect')),60000);const ready=e=>{if(e.detail==='ready'){clearTimeout(timer);$('avatar').removeEventListener('statechange',ready);resolve();}};$('avatar').addEventListener('statechange',ready);});playbackReady.catch(()=>{});
 session=await request('/v1/sessions',{method:'POST',body:form});
 $('avatar').session={id:session.id,url:location.origin,token:session.token};$('avatar').hidden=false;$('empty').hidden=true;$('portrait-preview').hidden=true;$('stop').hidden=false;
 const source=$('source').value;
 await connect(`/v1/sessions/${session.id}/${source==='file'?'audio':'demo/'+source}`,session.publisher_token);
 if(source==='file'){
   const file=$('audio').files[0];if(!file)throw new Error('Choose an audio file.');
   await playbackReady;
   $('status').textContent='Generating synchronized audio and video…';const audio=new FormData();audio.set('audio',file);await request(`/v1/sessions/${session.id}/file`,{method:'POST',headers:{Authorization:`Bearer ${session.publisher_token}`},body:audio});await send('end_turn');$('status').textContent='Playback complete. Start another session to try a new voice.';
 }else{
   media=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true}});context=new AudioContext({sampleRate:24000});await context.audioWorklet.addModule('/static/microphone.js');worklet=new AudioWorkletNode(context,'pcm-capture');context.createMediaStreamSource(media).connect(worklet);worklet.port.onmessage=({data})=>{if(ws?.readyState===WebSocket.OPEN&&ws.bufferedAmount<100000)ws.send(data);};worklet.connect(context.destination);$('status').textContent='Connected. Speak naturally; interrupt at any time.';
 }
}catch(error){$('status').textContent=error.message;await stop(false);}finally{$('start').disabled=false;}};
async function stop(message=true){media?.getTracks().forEach(t=>t.stop());worklet?.disconnect();await context?.close();context=null;ws?.close();if(session){await request(`/v1/sessions/${session.id}`,{method:'DELETE',headers:{Authorization:`Bearer ${session.publisher_token}`}}).catch(()=>{});session=null;}$('avatar').remove();const el=document.createElement('openspline-avatar');el.id='avatar';el.hidden=true;$('preview').append(el);$('portrait-preview').hidden=!previewURL;$('stop').hidden=true;if(message)$('status').textContent='Session ended.';}
$('stop').onclick=()=>void stop();window.addEventListener('pagehide',()=>{media?.getTracks().forEach(t=>t.stop());ws?.close();});
