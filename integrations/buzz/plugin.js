// Self-contained Buzz API v1 plugin. No local processes, files, or client MCP tools.
export const inject = ["agents", "react"];
const PROTOCOL = "missbot.buzz.v1";
const STORAGE_KEY = "missbot.hosted.v1";
const DEFAULT_ENDPOINT = "wss://missbot-buzz.taile6e57.ts.net/buzz/ws";
const HEX = /^[0-9a-f]{64}$/i;
const MAX_FRAME = 1024 * 1024;

export function remoteSettings(endpoint, token) {
  let url;
  try { url = new URL(endpoint); } catch { throw new Error("Enter a hosted WebSocket URL, such as wss://missbot.example/buzz/ws."); }
  const loopback = ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname);
  if (url.protocol !== "wss:" && !(url.protocol === "ws:" && loopback)) {
    throw new Error("Use wss:// for remote connections. ws:// is allowed only on loopback.");
  }
  if (url.username || url.password || url.search || url.hash) {
    throw new Error("Keep credentials, query parameters, and fragments out of the endpoint URL.");
  }
  if (typeof token !== "string" || !token.trim() || token.length > 2048) {
    throw new Error("Enter the hosted server's management token (at most 2048 characters).");
  }
  return { endpoint: url.href, token: token.trim() };
}

function managementRequest(settings, method, params, signal) {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(settings.endpoint, PROTOCOL);
    let authenticated = false, settled = false;
    const finish = (error, result) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal.removeEventListener("abort", abort);
      socket.onopen = socket.onmessage = socket.onerror = socket.onclose = null;
      if (socket.readyState < 2) socket.close();
      if (error) reject(error); else resolve(result);
    };
    const abort = () => finish(signal.reason ?? new Error("Hosted Missbot request cancelled"));
    const timer = setTimeout(() => finish(new Error(`Hosted Missbot ${method} timed out`)), 60000);
    signal.addEventListener("abort", abort, { once: true });
    const send = (message) => {
      const frame = JSON.stringify(message);
      if (frame.length * 3 + socket.bufferedAmount > MAX_FRAME) throw new Error("Hosted Missbot request is too large");
      socket.send(frame);
    };
    socket.onopen = () => {
      try {
        if (socket.protocol !== PROTOCOL) throw new Error("Hosted Missbot management protocol is unavailable");
        send({ type: "authenticate", token: settings.token });
      } catch (error) { finish(error); }
    };
    socket.onmessage = ({ data }) => {
      try {
        if (typeof data !== "string" || data.length > MAX_FRAME) throw new Error("Invalid hosted Missbot response");
        const message = JSON.parse(data);
        if (!authenticated) {
          if (message?.type !== "authenticated") throw new Error("Hosted Missbot authentication failed");
          authenticated = true;
          send({ jsonrpc: "2.0", id: 1, method, params });
          return;
        }
        if (!message || message.jsonrpc !== "2.0" || message.id !== 1 || message.method) {
          throw new Error("Invalid hosted Missbot response");
        }
        if (message.error) {
          const detail = typeof message.error.message === "string"
            ? `: ${message.error.message.replaceAll(settings.token, "[redacted]").slice(0, 300)}` : "";
          throw new Error(`Hosted Missbot ${method} failed (${message.error.code})${detail}`);
        }
        if (!message.result || typeof message.result !== "object" || Array.isArray(message.result)) {
          throw new Error("Invalid hosted Missbot result");
        }
        finish(null, message.result);
      } catch (error) { finish(error); }
    };
    socket.onerror = () => finish(new Error("Cannot connect to hosted Missbot; check its URL, TLS, and availability"));
    socket.onclose = ({ code }) => finish(new Error(code === 1008
      ? "Hosted Missbot authentication failed; check the management token"
      : "Hosted Missbot connection closed before the operation completed"));
    if (signal.aborted) abort();
  });
}

function connectionSetup(React, remote, lifetime) {
  return function ConnectionSetup({ disabled = false }) {
    const [endpoint, setEndpoint] = React.useState(remote.settings?.endpoint ?? DEFAULT_ENDPOINT);
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
          h("h3", { className: "buzz-content-header-title" }, "Hosted Missbot"),
          h("p", { className: "buzz-content-header-subtitle" }, "Missbot runs on its server, even when Buzz is closed. No desktop runner is needed."),
        ),
      ),
      h("div", { className: "buzz-settings-group", "data-buzz-ui": "", "data-layout": "form" },
        field("endpoint", "WebSocket endpoint", endpoint, setEndpoint, "url", DEFAULT_ENDPOINT),
        field("token", "Management token", token, setToken, "password", "Hosted server management token",
          "Saved unencrypted in this device's localStorage across plugin updates and restarts, never in shared agent configuration or URLs. This is not the OS keychain. ACP credentials are separate."),
        h("p", { className: "buzz-field-description m-0" },
          "Uses Grok's existing Nostr identity. Create a hosted agent; existing device-local agents are not migrated. Removing the hosted agent only deactivates Buzz, never Grok's shared key."),
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
            const status = await remote.connect(endpoint, token, signal);
            if (!signal.aborted) setMessage(`Service connected: ${status.pubkey}. ` +
              `${status.communities.filter((row) => row.active).length} active communities, ` +
              `${status.communities.filter((row) => row.connected).length} connected.` +
              status.communities.filter((row) => row.error).map((row) => ` ${row.community}: ${row.error}`).join(""));
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

export function apply(ctx) {
  const lifetime = new AbortController();
  const remote = {
    settings: null,
    async connect(endpoint, token, signal) {
      const settings = remoteSettings(endpoint, token);
      const status = await managementRequest(settings, "status", {}, signal);
      if (!HEX.test(status.pubkey ?? "") || !Array.isArray(status.communities) ||
          status.communities.some((row) => !row || typeof row.community !== "string" ||
            typeof row.name !== "string" || typeof row.active !== "boolean" ||
            typeof row.connected !== "boolean" || (row.error != null && typeof row.error !== "string"))) {
        throw new Error("Invalid hosted Missbot status");
      }
      signal.throwIfAborted();
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
      sessionStorage.removeItem(STORAGE_KEY);
      remote.settings = settings;
      return status;
    },
  };
  try {
    const persistent = localStorage.getItem(STORAGE_KEY);
    const saved = JSON.parse(persistent ?? sessionStorage.getItem(STORAGE_KEY) ?? "null");
    if (saved) {
      const settings = remoteSettings(saved.endpoint, saved.token);
      // Remove the session copy only after the durable write succeeds.
      if (persistent === null) localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
      sessionStorage.removeItem(STORAGE_KEY);
      remote.settings = settings;
    }
  } catch { /* Missing or invalid device settings require an explicit Connect. */ }
  const Setup = connectionSetup(ctx.react, remote, lifetime.signal);
  async function request(method, params, signal) {
    const combined = AbortSignal.any([signal, lifetime.signal]);
    combined.throwIfAborted();
    if (!remote.settings) throw new Error("Set the hosted WebSocket endpoint and management token in Missbot Settings, then Connect.");
    return managementRequest(remote.settings, method, params, combined);
  }
  ctx.agents.register({
    id: "missbot", title: "Missbot",
    description: "Hosted Grok identity: runs with Buzz closed. Existing local agents must be recreated.",
    defaults: () => ({ config: {} }), Create: Setup,
    tabs: [{ id: "connection", title: "Settings", role: "settings", component: Setup }],
    protectionUnavailable: "Missbot uses its server-configured tools and cannot enforce Buzz protection policies.",
    remote: {
      async provision({ name, picture, community, signal }) {
        const result = await request("provision", { name, picture, community }, signal);
        if (!HEX.test(result.pubkey ?? "") || (result.picture !== undefined && typeof result.picture !== "string")) {
          throw new Error("Invalid hosted Missbot identity");
        }
        return { pubkey: result.pubkey, ...(result.picture === undefined ? {} : { picture: result.picture }) };
      },
      async activate({ pubkey, name, community, auth, signal }) {
        await request("activate", { pubkey, name, community, auth }, signal);
      },
      async profile({ pubkey, community, name, picture, signal }) {
        await request("profile", { pubkey, community, name, picture }, signal);
      },
      async remove({ pubkey, community, signal }) {
        await request("remove", { pubkey, community }, signal);
      },
      async interrupt({ pubkey, community, channelId, threadRootId, signal }) {
        const result = await request("interrupt", { pubkey, community, channelId, threadRootId }, signal);
        if (typeof result.interrupted !== "boolean") throw new Error("Invalid hosted Missbot interruption result");
        return result.interrupted;
      },
    },
  });
  ctx.effect(() => () => lifetime.abort(new Error("Missbot plugin stopped")));
}
