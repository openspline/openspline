import { voice } from "@livekit/agents";
import { TrackKind, type Room } from "@livekit/rtc-node";
import { AccessToken } from "livekit-server-sdk";
import { Openspline, type AvatarSession, type Quality } from "./index.js";

/** Legacy transport options: url/apiKey/apiSecret refer to LiveKit. */
export interface LiveKitAvatarOptions {
  portrait?: string | Uint8Array;
  avatarId?: string;
  background?: string | Uint8Array;
  quality?: Quality;
  client?: Openspline;
  url?: string;
  apiKey?: string;
  apiSecret?: string;
  /** Video publication timeout in milliseconds, after worker preparation. */
  joinTimeout?: number;
}

/** Single-use native LiveKit avatar; the GPU worker publishes room tracks. */
export class LiveKitAvatarSession extends voice.AvatarSession {
  private avatar?: AvatarSession;
  private readonly identity = "openspline-" + crypto.randomUUID();
  private readonly client: Openspline;
  private readonly joinTimeout: number;
  private startPromise?: Promise<void>;
  private closePromise?: Promise<void>;
  private cleanupPromise?: Promise<void>;
  private closing = false;
  private session?: voice.AgentSession;
  private output?: voice.DataStreamAudioOutput;
  private previousOutput?: voice.AgentSession["output"]["audio"];

  constructor(private readonly options: LiveKitAvatarOptions) {
    super();
    this.client = options.client ?? new Openspline();
    if ((options.portrait === undefined) === (options.avatarId === undefined))
      throw new Error("Provide exactly one portrait or avatarId");
    if (
      options.avatarId !== undefined &&
      (!this.client.config.apiKey ||
        typeof options.avatarId !== "string" ||
        !options.avatarId)
    )
      throw new Error(
        "avatarId requires a Cloud API key and a saved avatar ID",
      );
    if (
      options.quality !== undefined &&
      !["low", "high"].includes(options.quality)
    )
      throw new Error("quality must be low or high");
    this.joinTimeout = options.joinTimeout ?? 30000;
    if (!Number.isFinite(this.joinTimeout) || this.joinTimeout <= 0)
      throw new Error(
        "joinTimeout must be a finite positive number of milliseconds",
      );
  }

  override get avatarIdentity() {
    return this.identity;
  }
  override get provider() {
    return "openspline";
  }

  override async start(
    agentSession: voice.AgentSession,
    room: Room,
  ): Promise<void> {
    if (this.closing || this.startPromise)
      throw new Error("AvatarSession is single-use; create a new session");
    if (!room.isConnected || !room.localParticipant?.identity || !room.name)
      throw new Error(
        "Connect the agent to its room before starting the avatar",
      );
    const url = this.options.url ?? process.env.LIVEKIT_URL;
    let parsed: URL;
    try {
      parsed = new URL(url!);
    } catch {
      throw new Error("Set LIVEKIT_URL to a ws:// or wss:// room server URL");
    }
    if (
      !["ws:", "wss:"].includes(parsed.protocol) ||
      !parsed.hostname ||
      parsed.username ||
      parsed.password ||
      parsed.search ||
      parsed.hash
    )
      throw new Error("Set LIVEKIT_URL to a ws:// or wss:// room server URL");
    const key = this.options.apiKey ?? process.env.LIVEKIT_API_KEY;
    const secret = this.options.apiSecret ?? process.env.LIVEKIT_API_SECRET;
    if (!key || !secret)
      throw new Error("Set LIVEKIT_API_KEY and LIVEKIT_API_SECRET");
    this.startPromise = this.startWorker(agentSession, room, url!, key, secret);
    await this.startPromise;
  }

  private checkClosing() {
    if (this.closing) throw new Error("AvatarSession closed during startup");
  }

  private async startWorker(
    session: voice.AgentSession,
    room: Room,
    url: string,
    key: string,
    secret: string,
  ) {
    try {
      await super.start(session, room);
      this.session = session;
      const token = new AccessToken(key, secret, {
        identity: this.identity,
        attributes: { "lk.publish_on_behalf": room.localParticipant!.identity },
      });
      token.kind = "agent";
      token.addGrant({ roomJoin: true, room: room.name });
      const jwt = await token.toJwt();
      this.checkClosing();
      this.avatar = await this.client.avatar(
        this.options.avatarId !== undefined
          ? { avatarId: this.options.avatarId }
          : this.options.portrait!,
        { quality: this.options.quality, background: this.options.background },
      );
      this.checkClosing();
      await this.avatar.attachLiveKit({
        url,
        token: jwt,
        sender_identity: room.localParticipant!.identity,
      });
      this.checkClosing();
      await this.waitForJoin({ timeout: this.joinTimeout });
      this.checkClosing();
      const participant = room.remoteParticipants.get(this.identity);
      if (
        !room.isConnected ||
        !participant ||
        ![...participant.trackPublications.values()].some(
          (p) => p.kind === TrackKind.KIND_VIDEO,
        )
      )
        throw new Error("Avatar disconnected before publishing video");
      this.previousOutput = session.output.audio;
      this.output = new voice.DataStreamAudioOutput({
        room,
        destinationIdentity: this.identity,
        waitPlaybackStart: true,
      });
      session.output.audio = this.output;
    } catch (error) {
      this.closing = true;
      await this.cleanup().catch(() => {});
      throw error;
    }
  }

  private cleanup(): Promise<void> {
    return (this.cleanupPromise ??= (async () => {
      const avatar = this.avatar;
      this.avatar = undefined;
      try {
        if (
          this.session &&
          this.output &&
          this.session.output.audio === this.output
        )
          this.session.output.audio = this.previousOutput ?? null;
      } finally {
        this.session = undefined;
        this.output = undefined;
        try {
          await avatar?.close();
        } finally {
          await super.aclose();
        }
      }
    })());
  }

  /** Release resources after any in-flight bounded provisioning request settles. */
  override async aclose(): Promise<void> {
    this.closing = true;
    this.closePromise ??= (async () => {
      await this.startPromise?.catch(() => {});
      await this.cleanup();
    })();
    await this.closePromise;
  }
}
