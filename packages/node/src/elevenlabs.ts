import type { AvatarSession } from './index.js';

function audioRate(format: string): number {
  if (format === 'ulaw_8000') return 8000;
  if (!/^pcm_(8000|16000|22050|24000|44100|48000)$/.test(format)) {
    throw new Error('Unsupported ElevenLabs audio format; use PCM or ulaw_8000.');
  }
  return Number(format.slice(4));
}

function decodeAudio(encoded: string, format: string): Buffer {
  const raw = Buffer.from(encoded, 'base64');
  if (format !== 'ulaw_8000') return raw;
  const pcm = Buffer.alloc(raw.length * 2);
  for (let i = 0; i < raw.length; i++) {
    const byte = ~raw[i] & 255;
    const sample = (((byte & 15) << 3) + 132) << ((byte >> 4) & 7);
    pcm.writeInt16LE(byte & 128 ? 132 - sample : sample - 132, i * 2);
  }
  return pcm;
}

/** Feed decoded WebSocket events, including initiation metadata. The app owns input, pings, and tools. */
export class ElevenLabsAgents {
  private sampleRate: number;
  private interruptedId = -1;
  private pending = false;

  constructor(private avatar: AvatarSession, private audioFormat = 'pcm_16000') {
    this.sampleRate = audioRate(audioFormat);
  }

  async handle(event: any): Promise<void> {
    if (event.type === 'conversation_initiation_metadata') {
      this.audioFormat = event.conversation_initiation_metadata_event.agent_output_audio_format;
      this.sampleRate = audioRate(this.audioFormat);
    } else if (event.type === 'interruption') {
      this.interruptedId = Math.max(this.interruptedId, event.interruption_event?.event_id ?? -1);
      this.pending = false;
      await this.avatar.interrupt();
    } else if (event.type === 'audio') {
      const audio = event.audio_event;
      if ((audio.event_id ?? 0) <= this.interruptedId) return;
      const raw = decodeAudio(audio.audio_base_64, this.audioFormat);
      if (raw.length) {
        await this.avatar.sendAudio(raw, { sample_rate: this.sampleRate });
        this.pending = true;
      }
      if (audio.is_final) await this.finish();
    } else if (event.type === 'agent_response_complete') {
      if ((event.agent_response_complete_event?.event_id ?? 0) > this.interruptedId) await this.finish();
    }
  }

  async *wrap(events: AsyncIterable<any>): AsyncGenerator<any> {
    for await (const event of events) {
      await this.handle(event);
      yield event;
    }
  }

  private async finish(): Promise<void> {
    if (this.pending) {
      await this.avatar.endTurn(false);
      this.pending = false;
    }
  }
}
