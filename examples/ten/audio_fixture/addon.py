import asyncio,json,os,wave
from ten_runtime import Addon,AsyncExtension,AudioFrame,Data,register_addon_as_extension
class Fixture(AsyncExtension):
    async def on_data(self,env,data):
        if data.get_name()!='avatar_session':return
        raw,error=data.get_property_to_json('session');session=json.loads(raw)
        print('Open viewer:',session['viewer_url'],flush=True)
        # Give the viewer time to connect. The avatar service bounds queued audio.
        self.task=asyncio.create_task(self.play(env))
    async def play(self,env):
        await asyncio.sleep(15)
        with wave.open(os.environ['AUDIO_WAV']) as wav:
            if wav.getsampwidth()!=2:raise ValueError('Use PCM16 WAV')
            while raw:=wav.readframes(wav.getframerate()//10):
                frame=AudioFrame.create('pcm_frame');frame.set_sample_rate(wav.getframerate());frame.set_number_of_channels(wav.getnchannels());frame.set_bytes_per_sample(2);frame.set_samples_per_channel(len(raw)//2//wav.getnchannels());frame.alloc_buf(len(raw));buf=frame.lock_buf();buf[:]=raw;frame.unlock_buf(buf)
                await env.send_audio_frame(frame)
        await env.send_data(Data.create('tts_audio_end'))
    async def on_stop(self,env):
        if hasattr(self,'task'):self.task.cancel();await asyncio.gather(self.task,return_exceptions=True)
@register_addon_as_extension('openspline_audio_fixture')
class FixtureAddon(Addon):
    def on_create_instance(self,env,name,context):env.on_create_instance_done(Fixture(name),context)
