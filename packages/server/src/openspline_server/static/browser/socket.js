import { nextPlayoutTime } from './playout.js';
/** PCM and JPEG share the Web Audio clock; feedback bounds the server's queue. */
export class SocketPlayback extends EventTarget {
    session;
    video;
    socket;
    context;
    canvas = document.createElement('canvas');
    stream;
    sources = new Set();
    frames = [];
    acknowledgments = [];
    heartbeat;
    timer;
    timeout;
    epoch = 0;
    end = 0;
    closed = false;
    initialized = false;
    originalMuted;
    portrait;
    idle = true;
    underruns = 0;
    lastSamples = 0;
    turnEnds = new Set();
    lastStats = 0;
    constructor(session, video) {
        super();
        this.session = session;
        this.video = video;
        this.originalMuted = video.muted;
    }
    event(type, detail) { this.dispatchEvent(new CustomEvent(type, { detail })); }
    send(event) { if (this.socket?.readyState === WebSocket.OPEN)
        this.socket.send(JSON.stringify(event)); }
    fail(message) { if (this.closed)
        return; this.close(); this.event('error', message); }
    async connect() {
        const url = new URL(`/v1/sessions/${this.session.id}/playback`, this.session.url);
        url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
        const socket = this.socket = new WebSocket(url);
        this.timeout = setTimeout(() => this.fail('Playback connection timed out. Check that your proxy forwards WebSockets.'), 15000);
        socket.onopen = () => this.send({ token: this.session.token });
        socket.onerror = () => this.fail('Playback WebSocket could not connect. Check the server or proxy.');
        socket.onclose = e => this.fail(e.reason || 'Playback connection closed. Start a new session.');
        let messages = Promise.resolve();
        socket.onmessage = ({ data }) => {
            // Preserve decode order. The server bounds outstanding audio to one second.
            messages = messages.then(() => this.receive(JSON.parse(data))).catch(error => this.fail(String(error)));
        };
    }
    async receive(event) {
        if (this.closed)
            return;
        if (event.type === 'connected') {
            clearTimeout(this.timeout);
            this.epoch = event.epoch;
            this.context = new AudioContext({ sampleRate: 48000 });
            this.canvas.width = this.canvas.height = 512;
            const response = await fetch(`${this.session.url}/v1/sessions/${this.session.id}/portrait`, { headers: { Authorization: `Bearer ${this.session.token}` } });
            if (!response.ok)
                throw new Error('Could not load avatar portrait');
            const image = await createImageBitmap(await response.blob());
            if (this.closed) {
                image.close();
                return;
            }
            this.portrait = image;
            this.canvas.getContext('2d').drawImage(image, 0, 0, 512, 512);
            this.stream = this.canvas.captureStream(25);
            this.video.srcObject = this.stream;
            // Only the Web Audio output plays sound; the video stream contains pictures.
            this.video.muted = true;
            this.video.playsInline = true;
            this.initialized = true;
            this.heartbeat = setInterval(() => this.send({ type: 'ping' }), 15000);
            this.timer = setInterval(() => this.tick(), 10);
            // Autoplay may require a click. Never mark playback ready before permission.
            if (this.context.state === 'suspended')
                this.event('autoplayblocked');
            else
                await this.play().catch(() => this.event('autoplayblocked'));
        }
        else if (event.type === 'media') {
            if (event.epoch !== this.epoch)
                return;
            const bytes = Uint8Array.from(atob(event.audio), (c) => c.charCodeAt(0));
            const imageBytes = Uint8Array.from(atob(event.image), (c) => c.charCodeAt(0));
            const image = await createImageBitmap(new Blob([imageBytes], { type: 'image/jpeg' }));
            if (this.closed || event.epoch !== this.epoch) {
                image.close();
                return;
            }
            const context = this.context, buffer = context.createBuffer(1, bytes.length / 2, 48000), samples = buffer.getChannelData(0), view = new DataView(bytes.buffer);
            for (let i = 0; i < samples.length; i++)
                samples[i] = view.getInt16(i * 2, true) / 32768;
            const now = context.currentTime;
            const at = nextPlayoutTime(now, this.end);
            if (this.end > 0 && at > this.end && !this.turnEnds.has(this.lastSamples))
                this.underruns++;
            this.lastSamples = event.samples;
            for (const end of this.turnEnds)
                if (end < event.samples)
                    this.turnEnds.delete(end);
            const source = context.createBufferSource();
            source.buffer = buffer;
            source.connect(context.destination);
            this.sources.add(source);
            source.onended = () => { this.sources.delete(source); source.disconnect(); };
            source.start(at);
            this.end = at + buffer.duration;
            this.idle = false;
            const latency = context.outputLatency || context.baseLatency || 0;
            this.frames.push({ at: at + latency, image });
            this.acknowledgments.push({ at: this.end + latency, samples: event.samples, epoch: event.epoch });
        }
        else {
            if (event.type === 'interrupted') {
                this.epoch = event.epoch;
                this.clear();
                this.event('statechange', 'ready');
            }
            if (event.type === 'playback_ready')
                this.event('statechange', 'ready');
            if (event.type === 'speaking')
                this.event('statechange', 'speaking');
            if (event.type === 'turn_end') {
                this.turnEnds.add(event.target_samples);
                this.event('turnend', event);
            }
            if (event.type === 'error')
                this.fail(event.message);
            this.event('event', event);
        }
    }
    tick() {
        const context = this.context;
        if (!context || context.state !== 'running')
            return;
        if (context.currentTime - this.lastStats >= 1) {
            this.lastStats = context.currentTime;
            this.send({ type: 'playback_stats', epoch: this.epoch, buffer_ms: Math.round(Math.max(0, this.end - context.currentTime) * 1000), underruns: this.underruns });
        }
        while (this.frames.length && this.frames[0].at <= context.currentTime) {
            const { image } = this.frames.shift();
            this.canvas.getContext('2d').drawImage(image, 0, 0, 512, 512);
            image.close();
        }
        while (this.acknowledgments.length && this.acknowledgments[0].at <= context.currentTime) {
            const { samples, epoch } = this.acknowledgments.shift();
            this.send({ type: 'played', samples, epoch });
        }
        if (!this.idle && !this.frames.length && !this.acknowledgments.length && this.portrait) {
            this.canvas.getContext('2d').drawImage(this.portrait, 0, 0, 512, 512);
            this.idle = true;
        }
    }
    async play() {
        if (!this.initialized || this.closed)
            return;
        await this.context.resume();
        await this.video.play();
        if (this.context.state !== 'running')
            throw new Error('Enable audio playback to continue');
        this.send({ type: 'ready' });
    }
    clear() {
        for (const source of this.sources) {
            source.stop();
            source.disconnect();
        }
        this.sources.clear();
        for (const frame of this.frames)
            frame.image.close();
        this.frames = [];
        this.acknowledgments = [];
        this.end = 0;
        this.lastSamples = 0;
        this.turnEnds.clear();
        if (this.portrait)
            this.canvas.getContext('2d').drawImage(this.portrait, 0, 0, 512, 512);
    }
    close() {
        if (this.closed)
            return;
        this.closed = true;
        clearTimeout(this.timeout);
        clearInterval(this.heartbeat);
        clearInterval(this.timer);
        this.clear();
        this.socket?.close();
        void this.context?.close();
        this.portrait?.close();
        this.portrait = undefined;
        this.stream?.getTracks().forEach(track => track.stop());
        this.video.srcObject = null;
        this.video.muted = this.originalMuted;
    }
}
