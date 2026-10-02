export class AvatarConnection extends EventTarget {
    session;
    video;
    pc;
    channel;
    timers = new Set();
    heartbeat;
    statsTimer;
    generation = 0;
    epoch = 0;
    lastSamples = -1;
    playoutDelay = 150;
    stream;
    aborted = false;
    state = 'idle';
    constructor(session, video) {
        super();
        this.session = session;
        this.video = video;
    }
    status(state) { this.state = state; this.dispatchEvent(new CustomEvent('statechange', { detail: state })); }
    async connect() {
        this.aborted = false;
        this.status('connecting');
        const generation = ++this.generation;
        const ice = await fetch(`${this.session.url}/v1/sessions/${this.session.id}/ice`, { headers: { Authorization: `Bearer ${this.session.token}` } });
        if (!ice.ok)
            throw new Error('Playback credentials expired or session unavailable');
        const config = await ice.json();
        if (this.aborted || generation !== this.generation)
            return;
        const pc = this.pc = new RTCPeerConnection(config);
        this.stream = new MediaStream();
        this.video.srcObject = this.stream;
        this.video.playsInline = true;
        pc.addTransceiver('audio', { direction: 'recvonly' });
        pc.addTransceiver('video', { direction: 'recvonly' });
        pc.ontrack = e => { this.stream.addTrack(e.track); };
        pc.onconnectionstatechange = () => { if (pc.connectionState === 'failed' || pc.connectionState === 'closed')
            this.status('disconnected'); };
        this.channel = pc.createDataChannel('openspline');
        this.channel.onopen = () => { if (!this.video.paused)
            this.channel?.send(JSON.stringify({ type: 'ready' })); this.status('ready'); };
        this.channel.onmessage = e => {
            const event = JSON.parse(e.data);
            this.epoch = event.epoch ?? this.epoch;
            if (event.type === 'clock' && event.samples !== this.lastSamples) {
                this.lastSamples = event.samples;
                const timer = setTimeout(() => { this.timers.delete(timer); if (event.epoch === this.epoch && !this.video.paused && this.channel?.readyState === 'open')
                    this.channel.send(JSON.stringify({ type: 'played', samples: event.samples, epoch: event.epoch })); }, this.playoutDelay);
                this.timers.add(timer);
            }
            if (event.type === 'speaking')
                this.status('speaking');
            if (event.type === 'turn_end') {
                this.dispatchEvent(new CustomEvent('turnend', { detail: event }));
            }
            if (event.type === 'interrupted') {
                for (const timer of this.timers)
                    clearTimeout(timer);
                this.timers.clear();
                this.lastSamples = -1;
                this.status('ready');
                // Silence the browser immediately while stale RTP packets drain.
                const muted = this.video.muted;
                this.video.muted = true;
                const timer = setTimeout(() => { this.video.muted = muted; this.timers.delete(timer); }, this.playoutDelay + 60);
                this.timers.add(timer);
            }
            if (event.type === 'error') {
                this.status('error');
                this.dispatchEvent(new CustomEvent('error', { detail: event.message }));
            }
            this.dispatchEvent(new CustomEvent('event', { detail: event }));
        };
        try {
            await pc.setLocalDescription(await pc.createOffer());
            if (pc.iceGatheringState !== 'complete')
                await new Promise((resolve, reject) => { const timer = setTimeout(() => reject(new Error('ICE gathering timed out')), 15000); pc.addEventListener('icegatheringstatechange', () => { if (pc.iceGatheringState === 'complete') {
                    clearTimeout(timer);
                    resolve();
                } }); });
            if (this.aborted || generation !== this.generation) {
                pc.close();
                return;
            }
            const response = await fetch(`${this.session.url}/v1/sessions/${this.session.id}/offer`, { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${this.session.token}` }, body: JSON.stringify(pc.localDescription) });
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error?.message ?? 'Cannot connect to avatar');
            }
            if (this.aborted || generation !== this.generation) {
                pc.close();
                return;
            }
            await pc.setRemoteDescription(await response.json());
            this.heartbeat = setInterval(() => { if (this.channel?.readyState === 'open')
                this.channel.send(JSON.stringify({ type: 'ping' })); }, 15000);
            this.statsTimer = setInterval(async () => { if (pc.connectionState !== 'connected')
                return; const reports = await pc.getStats(); reports.forEach(report => { if (report.type === 'inbound-rtp' && report.kind === 'audio' && report.jitterBufferEmittedCount)
                this.playoutDelay = Math.min(500, Math.max(60, 1000 * report.jitterBufferDelay / report.jitterBufferEmittedCount + 40)); }); }, 2000);
            await this.play().catch(() => { this.dispatchEvent(new Event('autoplayblocked')); });
        }
        catch (error) {
            this.close();
            this.status('error');
            throw error;
        }
    }
    async play() { await this.video.play(); this.lastSamples = -1; if (this.channel?.readyState === 'open')
        this.channel.send(JSON.stringify({ type: 'ready' })); }
    close() { this.aborted = true; ++this.generation; for (const timer of this.timers)
        clearTimeout(timer); this.timers.clear(); clearInterval(this.heartbeat); clearInterval(this.statsTimer); this.channel?.close(); this.pc?.close(); this.video.srcObject = null; this.status('disconnected'); }
}
