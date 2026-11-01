// Self-contained Buzz API v1 plugin. No local processes, files, or client MCP tools.
export const inject = ["agents", "react"];
const PROTOCOL = "missbot.acp.v1";
const STORAGE_KEY = "missbot.remote.v1";
const HEX = /^[0-9a-f]{64}$/i;
const MAX_TEXT = 65536;
const MAX_FRAME = 1024 * 1024;
const MAX_CONVERSATIONS = 8;
const DEADLINE = 300000;

function abortable(promise, signal) {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const abort = () => reject(signal.reason);
    signal.addEventListener("abort", abort, { once: true });
    promise.then(resolve, reject).finally(() => signal.removeEventListener("abort", abort));
  });
}

export function remoteSettings(endpoint, token) {
  let url;
  try { url = new URL(endpoint); } catch { throw new Error("Enter a remote WebSocket URL, such as wss://missbot.example/acp/ws."); }
  const loopback = ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname);
  if (url.protocol !== "wss:" && !(url.protocol === "ws:" && loopback)) {
    throw new Error("Use wss:// for remote connections. ws:// is allowed only on loopback.");
  }
  if (url.username || url.password || url.search || url.hash) {
    throw new Error("Keep credentials, query parameters, and fragments out of the endpoint URL.");
  }
  if (typeof token !== "string" || !token.trim() || token.length > 2048) {
    throw new Error("Enter the remote server's token (at most 2048 characters).");
  }
  return { endpoint: url.href, token: token.trim() };
}

export class RemoteConnection {
  constructor(settings) {
    this.settings = settings;
    this.pending = new Map();
    this.sequence = 0;
    this.closed = false;
    this.text = "";
    this.authenticated = false;
  }

  async open(signal) {
    signal.throwIfAborted();
    const abort = () => this.close(new Error("Remote Missbot connection cancelled"));
    signal.addEventListener("abort", abort, { once: true });
    const timer = setTimeout(() => this.close(new Error("Remote Missbot connection timed out")), 15000);
    try {
      await new Promise((resolve, reject) => {
        this.rejectOpening = reject;
        const socket = this.socket = new WebSocket(this.settings.endpoint, PROTOCOL);
        socket.onopen = () => {
          if (socket.protocol !== PROTOCOL) {
            this.close(new Error("Update the remote ACP server: browser authentication is unavailable"));
            return;
          }
          try { this.send({ type: "authenticate", token: this.settings.token }); }
          catch { this.close(new Error("Cannot authenticate with remote Missbot")); }
        };
        socket.onmessage = ({ data }) => {
          if (this.authenticated) { this.read(data); return; }
          try {
            if (typeof data !== "string" || data.length > MAX_FRAME || JSON.parse(data).type !== "authenticated") {
              throw new Error("Invalid authentication response");
            }
            this.authenticated = true;
            resolve();
          } catch { this.close(new Error("Remote Missbot authentication failed")); }
        };
        socket.onerror = () => this.close(new Error("Cannot connect to remote Missbot; check its URL, TLS, and availability"));
        socket.onclose = ({ code }) => this.close(new Error(code === 1008
          ? "Remote Missbot authentication failed; check the token"
          : "Remote Missbot connection closed"));
      });
      const init = await this.request("initialize", {
        protocolVersion: 1, clientCapabilities: {}, clientInfo: { name: "missbot-buzz-app", version: "1" },
      });
      if (init.protocolVersion !== 1) throw new Error("Missbot requires ACP protocol version 1");
      const session = await this.request("session/new", {
        cwd: "/", mcpServers: [], _meta: { missbot: { private: true } },
      });
      if (typeof session.sessionId !== "string" || session._meta?.missbot?.private !== true) {
        throw new Error("Update the remote ACP server: it must acknowledge private sessions");
      }
      this.session = session.sessionId;
    } catch (error) {
      this.close(error);
      throw error;
    } finally {
      clearTimeout(timer);
      signal.removeEventListener("abort", abort);
    }
  }

  send(message) {
    const frame = JSON.stringify(message);
    if (this.closed || this.socket?.readyState !== 1) throw new Error("Remote Missbot connection is closed");
    if (this.socket.bufferedAmount + frame.length * 3 > MAX_FRAME) throw new Error("Remote Missbot output backlog exceeded");
    this.socket.send(frame);
  }

  request(method, params, timeout = 60000) {
    if (this.closed) return Promise.reject(new Error("Remote Missbot connection is closed"));
    const id = ++this.sequence;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => this.close(new Error(`Missbot ACP ${method} timed out`)), timeout);
      this.pending.set(id, { resolve, reject, timer });
      try { this.send({ jsonrpc: "2.0", id, method, params }); }
      catch (error) { this.close(error); }
    });
  }

  read(frame) {
    if (this.closed) return;
    try {
      if (typeof frame !== "string" || frame.length > MAX_FRAME) throw new Error("Invalid or oversized ACP frame");
      const message = JSON.parse(frame);
      if (!message || message.jsonrpc !== "2.0") throw new Error("Invalid ACP frame");
      if (message.method) {
        if (message.id !== undefined) {
          this.send({ jsonrpc: "2.0", id: message.id, error: { code: -32601, message: "Client method not supported" } });
        } else if (message.method === "session/update" && message.params?.sessionId === this.session) {
          const update = message.params.update;
          if (update?.sessionUpdate === "agent_message_chunk" && update.content?.type === "text") {
            if (typeof update.content.text !== "string") throw new Error("Invalid ACP text chunk");
            this.text += update.content.text;
            if (this.text.length > MAX_TEXT) throw new Error("Missbot reply exceeds 65536 characters");
          }
        }
        return;
      }
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      clearTimeout(pending.timer);
      if (message.error) pending.reject(new Error(`Missbot ACP request failed (${message.error.code})`));
      else if (!message.result || typeof message.result !== "object") pending.reject(new Error("Invalid ACP result"));
      else pending.resolve(message.result);
    } catch (error) { this.close(error); }
  }

  async prompt(text) {
    this.text = "";
    const result = await this.request("session/prompt", {
      sessionId: this.session, prompt: [{ type: "text", text }],
    }, DEADLINE);
    if (["cancelled", "refusal"].includes(result.stopReason)) return "";
    if (result.stopReason !== "end_turn") throw new Error(`Missbot stopped without a complete reply (${result.stopReason})`);
    return this.text.trim() === "NO_REPLY" ? "" : this.text;
  }

  close(error = new Error("Remote Missbot conversation closed")) {
    if (this.closed) return;
    this.closed = true;
    this.rejectOpening?.(error);
    for (const { reject, timer } of this.pending.values()) { clearTimeout(timer); reject(error); }
    this.pending.clear();
    this.text = "";
    if (this.socket && this.socket.readyState < 2) this.socket.close();
  }
}

function connectionSetup(React, remote, lifetime) {
  return function ConnectionSetup({ disabled = false }) {
    const [endpoint, setEndpoint] = React.useState(remote.settings?.endpoint ?? "");
    const [token, setToken] = React.useState(remote.settings?.token ?? "");
    const [busy, setBusy] = React.useState(false);
    const [message, setMessage] = React.useState("");
    const [failed, setFailed] = React.useState(false);
    const operation = React.useRef();
    const id = React.useId();
    React.useEffect(() => () => operation.current?.abort(), []);
    const h = React.createElement;
    const unavailable = disabled || busy || !endpoint.trim() || !token.trim();
    // External plugins reuse the host's CSS roles, not private React module imports.
    const field = (name, label, value, change, type, placeholder, description) =>
      h("div", { className: "buzz-field", "data-buzz-ui": "" },
        h("label", { className: "buzz-field-label", htmlFor: `${id}-${name}` }, label),
        h("input", {
          id: `${id}-${name}`, className: "buzz-input", "data-buzz-ui": "", "data-size": "md",
          type, value, placeholder, disabled: disabled || busy, spellCheck: false, autoComplete: "off",
          "aria-describedby": description ? `${id}-${name}-help` : undefined,
          onChange: (event) => change(event.target.value),
        }),
        description && h("p", { id: `${id}-${name}-help`, className: "buzz-field-description m-0" }, description),
      );
    return h("section", { className: "grid gap-4", "data-buzz-ui": "" },
      h("div", { className: "buzz-content-header", "data-buzz-ui": "", "data-inline": true },
        h("div", { className: "buzz-content-header-copy" },
          h("h3", { className: "buzz-content-header-title" }, "Remote Missbot"),
          h("p", { className: "buzz-content-header-subtitle" }, "Connect directly to your running Missbot ACP server. Nothing is installed on this computer."),
        ),
      ),
      h("div", { className: "buzz-settings-group", "data-buzz-ui": "", "data-layout": "form" },
        field("endpoint", "WebSocket endpoint", endpoint, setEndpoint, "url", "wss://missbot.example/acp/ws"),
        field("token", "Access token", token, setToken, "password", "Server ACP token",
          "These settings apply to Missbot agents on this device. The token is kept only for this Buzz session, never in agent configuration or URLs. Re-enter it after restarting Buzz."),
        h("div", { className: "flex flex-wrap items-center gap-3" }, h("button", {
        className: "buzz-button", "data-buzz-ui": "", "data-variant": "prominent",
        "data-size": "md", "data-button-shape": "capsule",
        "data-disabled": unavailable ? "" : undefined, "data-loading": busy || undefined, "aria-busy": busy,
        type: "button", disabled: unavailable,
        onClick: async () => {
          setBusy(true); setMessage(""); setFailed(false);
          operation.current = new AbortController();
          const signal = AbortSignal.any([operation.current.signal, lifetime]);
          try {
            await remote.connect(endpoint, token, signal);
            if (!signal.aborted) setMessage("Connected. Send Missbot a new message.");
          } catch (error) {
            if (!signal.aborted) { setFailed(true); setMessage(error.message); }
          } finally { if (!signal.aborted) setBusy(false); }
        },
      }, h("span", { className: "buzz-button-label" }, busy ? "Connecting…" : "Connect"),
        busy && h("svg", {
          className: "buzz-button-spinner", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor",
          strokeWidth: 2, "aria-hidden": true, focusable: false,
        }, h("path", { d: "M20.66 7A10 10 0 1 1 7 3.34" })),
      )),
        message && h("p", {
          role: failed ? "alert" : "status", className: `${failed ? "buzz-field-error" : "buzz-field-description"} m-0`,
        }, message),
      ),
    );
  };
}

function eventPrompt(event) {
  if (event.kind !== 9 || !HEX.test(event.id) || !HEX.test(event.pubkey) ||
      !Number.isSafeInteger(event.created_at) || typeof event.content !== "string") {
    throw new Error("Missbot requires a Buzz kind-9 message with event identity and time");
  }
  const text = `Event ID: ${event.id.toLowerCase()}\nFrom: (hex: ${event.pubkey.toLowerCase()})\n` +
    `Time: ${new Date(event.created_at * 1000).toISOString()}\nContent: ${event.content}`;
  if (text.length > MAX_TEXT) throw new Error("Message exceeds Missbot's 65536-character input limit");
  return text;
}

async function destination(delivery) {
  const { event } = delivery.trigger;
  const channels = event.tags.filter(([name]) => name === "h").map(([, value]) => value);
  const channel = delivery.channelId;
  if (!channel || channels.length !== 1 || channels[0] !== channel) throw new Error("Cannot determine an unambiguous Buzz reply channel");
  const metadata = await abortable(delivery.agent.query([{ kinds: [39000], "#d": [channel], limit: 1 }]), delivery.signal);
  delivery.signal.throwIfAborted();
  const head = metadata.find((item) => item.kind === 39000 && item.tags.some(([name, value]) => name === "d" && value === channel));
  if (!head) throw new Error("Buzz channel metadata is unavailable; refusing to guess the reply destination");
  const types = head.tags.filter(([name]) => name === "t").map(([, value]) => value);
  const flat = types[0] === "dm" || (types[0] === "stream" && types.slice(1).includes("buzz.solo") && head.tags.some(([name]) => name === "private"));
  // Match Buzz's NIP-10 reader: last valid marked tag wins; a lone root is not a reply.
  let root, reply;
  for (const [name, value, , marker] of event.tags) {
    if (name !== "e" || !HEX.test(value ?? "")) continue;
    if (marker === "root") root = value.toLowerCase();
    if (marker === "reply") reply = value.toLowerCase();
  }
  root = flat ? undefined : reply ? root ?? reply : event.id.toLowerCase();
  return { channel, root };
}

export function apply(ctx) {
  const { agents } = ctx;
  const conversations = new Map();
  const completed = new Set();
  const lifetime = new AbortController();
  let queue = Promise.resolve();
  let waiting = 0;
  function forget(key) { conversations.get(key)?.connection.close(); conversations.delete(key); }
  const remote = {
    settings: null,
    async connect(endpoint, token, signal) {
      const settings = remoteSettings(endpoint, token);
      const probe = new RemoteConnection(settings);
      try { await probe.open(signal); } finally { probe.close(); }
      signal.throwIfAborted();
      // Session storage is device-local and is not included in shared agent config.
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
      for (const key of conversations.keys()) forget(key);
      remote.settings = settings;
    },
  };
  try {
    const saved = JSON.parse(sessionStorage.getItem(STORAGE_KEY) ?? "null");
    if (saved) remote.settings = remoteSettings(saved.endpoint, saved.token);
  } catch { /* Missing or invalid device settings require an explicit Connect. */ }
  const Setup = connectionSetup(ctx.react, remote, lifetime.signal);

  async function run(delivery) {
    if (delivery.trigger.type !== "mention" || delivery.trigger.event.pubkey === delivery.agent.pubkey) { delivery.ignored?.(); return; }
    if (!remote.settings) throw new Error("Set the remote WebSocket endpoint and access token in Missbot Settings, then Connect.");
    if (delivery.agent.protection) throw new Error("Missbot cannot enforce Buzz protected-agent policies");
    const id = JSON.stringify([delivery.agent.key, delivery.trigger.event.id]);
    if (completed.has(id)) { delivery.ignored?.(); return; }
    const text = eventPrompt(delivery.trigger.event);
    const controller = new AbortController();
    const signal = AbortSignal.any([delivery.signal, lifetime.signal, controller.signal, AbortSignal.timeout(DEADLINE)]);
    signal.throwIfAborted();
    delivery.interruptible?.(async () => controller.abort(new Error("Missbot turn stopped")));
    const { channel, root } = await destination({ ...delivery, signal });
    signal.throwIfAborted();
    const key = JSON.stringify([delivery.agent.key, channel, root ?? null]);
    let entry = conversations.get(key);
    if (entry?.connection.closed) { forget(key); entry = undefined; }
    if (!entry) {
      if (conversations.size >= MAX_CONVERSATIONS) forget(conversations.keys().next().value);
      entry = { owner: delivery.agent.key, connection: new RemoteConnection(remote.settings) };
      conversations.set(key, entry);
    } else { conversations.delete(key); conversations.set(key, entry); }
    const { connection } = entry;
    const abort = () => connection.close(new Error("Missbot turn cancelled"));
    signal.addEventListener("abort", abort, { once: true });
    try {
      if (!connection.session) await connection.open(signal);
      signal.throwIfAborted();
      delivery.acknowledged?.();
      const reply = await connection.prompt(text);
      signal.throwIfAborted();
      // Relay acceptance may precede a lost acknowledgement. Never automatically replay publication.
      completed.add(id);
      if (completed.size > 1024) completed.delete(completed.values().next().value);
      if (reply.trim()) await delivery.agent.publish({
        kind: 9, content: reply, tags: [["h", channel], ...(root ? [["e", root, "", "reply"]] : [])],
      }, () => signal.throwIfAborted());
    } catch (error) { forget(key); throw error; }
    finally { signal.removeEventListener("abort", abort); }
  }

  agents.register({
    id: "missbot", title: "Missbot",
    description: "Connect directly to a remote Missbot ACP server. No local runtime required.",
    defaults: () => ({ config: {} }), Create: Setup,
    tabs: [{ id: "connection", title: "Settings", role: "settings", component: Setup }],
    protectionUnavailable: "Missbot uses its server-configured tools and cannot enforce Buzz protection policies.",
    timeoutMs: DEADLINE,
    run(delivery) {
      if (lifetime.signal.aborted) return Promise.reject(new Error("Missbot plugin is stopped"));
      if (waiting >= 20) return Promise.reject(new Error("Missbot queue is full"));
      waiting++;
      // ponytail: serialize up to 20 turns globally; use per-conversation queues if throughput matters.
      const result = queue.then(() => run(delivery)).finally(() => waiting--);
      queue = result.catch(() => {});
      return result;
    },
    async retire(owner) { for (const [key, entry] of conversations) if (entry.owner === owner) forget(key); },
  });
  const unsubscribe = agents.subscribe(() => {
    const snapshot = agents.snapshot();
    if (snapshot.status === "loading") return;
    const running = new Set(snapshot.running.map((agent) => agent.key));
    for (const [key, entry] of conversations) if (!running.has(entry.owner)) forget(key);
  });
  ctx.effect(() => () => {
    lifetime.abort(new Error("Missbot plugin stopped"));
    unsubscribe();
    for (const key of conversations.keys()) forget(key);
    completed.clear();
  });
}
