import { AvatarConnection } from './index.js';
/** Explicit registration keeps imports safe in server rendering environments. */
export function registerAvatarElement(name = 'openspline-avatar') {
    if (typeof window === 'undefined' || customElements.get(name))
        return;
    class AvatarElement extends HTMLElement {
        connection;
        descriptor;
        video;
        status;
        button;
        constructor() {
            super();
            const shadow = this.attachShadow({ mode: 'open' });
            shadow.innerHTML = `<style>
      :host{display:block;--avatar-radius:20px;--avatar-background:#151715;--avatar-color:#eeeade;color:var(--avatar-color);font:13px ui-monospace,monospace}
      .frame{position:relative;aspect-ratio:1;background:var(--avatar-background);border-radius:var(--avatar-radius);overflow:hidden}video{width:100%;height:100%;object-fit:cover;display:block}.badge{position:absolute;left:16px;bottom:16px;background:#151715dd;padding:8px 12px;border-radius:30px;display:flex;gap:8px;align-items:center}.dot{width:6px;height:6px;border-radius:50%;background:#d3ee83}button{position:absolute;inset:0;margin:auto;width:160px;height:44px;border:0;border-radius:24px;background:#d3ee83;color:#172015;font:inherit;cursor:pointer}button[hidden]{display:none}video:focus-visible,button:focus-visible{outline:3px solid #d3ee83;outline-offset:-4px}
      </style><div class="frame"><video playsinline aria-label="Live avatar"></video><div class="badge"><i class="dot"></i><span role="status" aria-live="polite">Ready to connect</span></div><button hidden>Enable playback</button></div>`;
            this.video = shadow.querySelector('video');
            this.status = shadow.querySelector('[role=status]');
            this.button = shadow.querySelector('button');
            this.button.onclick = () => void this.connection?.play().then(() => { this.button.hidden = true; }).catch(() => { this.status.textContent = 'Playback is blocked'; });
        }
        set session(value) { this.descriptor = value; if (this.isConnected)
            void this.start(); }
        get session() { return this.descriptor; }
        connectedCallback() { if (this.descriptor)
            void this.start(); }
        disconnectedCallback() { this.connection?.close(); }
        async start() { this.connection?.close(); this.connection = new AvatarConnection(this.descriptor, this.video); this.connection.addEventListener('statechange', e => { this.status.textContent = e.detail; this.dispatchEvent(new CustomEvent('statechange', { detail: e.detail })); }); this.connection.addEventListener('autoplayblocked', () => { this.button.hidden = false; }); try {
            await this.connection.connect();
        }
        catch (error) {
            this.status.textContent = String(error);
            this.dispatchEvent(new CustomEvent('avatarerror', { detail: error }));
        } }
    }
    customElements.define(name, AvatarElement);
}
