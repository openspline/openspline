'use client';
import {useEffect,useRef,useState,type CSSProperties} from 'react';
import {AvatarConnection,type AvatarState,type SessionDescriptor} from '@openspline/browser';
export type {SessionDescriptor};
export function useAvatar(session:SessionDescriptor){
  const videoRef=useRef<HTMLVideoElement>(null);const connection=useRef<AvatarConnection|null>(null);
  const [state,setState]=useState<AvatarState>('idle');const [error,setError]=useState<string>();const [blocked,setBlocked]=useState(false);
  useEffect(()=>{if(!videoRef.current)return;const next=new AvatarConnection(session,videoRef.current);connection.current=next;setError(undefined);next.addEventListener('statechange',e=>setState((e as CustomEvent).detail));next.addEventListener('autoplayblocked',()=>setBlocked(true));next.connect().catch(e=>setError(String(e)));return()=>next.close();},[session.id,session.url,session.token]);
  return {videoRef,state,error,blocked,play:async()=>{await connection.current?.play();setBlocked(false);}};
}
export function Avatar({session,className,style,onStateChange}:{session:SessionDescriptor;className?:string;style?:CSSProperties;onStateChange?:(state:AvatarState)=>void}){
  const {videoRef,state,error,blocked,play}=useAvatar(session);useEffect(()=>onStateChange?.(state),[state,onStateChange]);
  return <div className={className} style={{position:'relative',aspectRatio:'1',overflow:'hidden',borderRadius:'var(--avatar-radius, 20px)',background:'var(--avatar-background, #151715)',color:'var(--avatar-color, #eeeade)',...style}}>
    <video ref={videoRef} playsInline aria-label="Live avatar" style={{display:'block',width:'100%',height:'100%',objectFit:'cover'}}/>
    <span role="status" aria-live="polite" style={{position:'absolute',bottom:16,left:16,padding:'8px 12px',borderRadius:24,background:'#151715dd',font:'12px monospace'}}>{error??state}</span>
    {blocked&&<button onClick={()=>void play()} style={{position:'absolute',top:'45%',left:'25%',width:'50%',padding:12,border:0,borderRadius:24,background:'#d3ee83',color:'#172015'}}>Enable playback</button>}
  </div>;
}
