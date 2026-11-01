import assert from "node:assert/strict";
import { test } from "node:test";
import { apply, remoteSettings } from "./plugin.js";

const OriginalWebSocket = globalThis.WebSocket;
const OriginalStorage = Object.getOwnPropertyDescriptor(globalThis, "sessionStorage");
const OriginalLocalStorage = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
const STORAGE_KEY = "missbot.hosted.v1";
const bot = "b".repeat(64);
const community = "wss://community.example/";
const settings = { endpoint: "wss://missbot.example/buzz/ws", token: "private-token" };
const input = () => ({ name: "Grok", community, signal: new AbortController().signal });

function harness(t, { authenticated = true, configured = true, persistent = new Map(),
  storageFailure = false, session, protocol = "missbot.buzz.v1", respond } = {}) {
  const sockets = [];
  const storage = session ?? new Map(configured ? [[STORAGE_KEY, JSON.stringify(settings)]] : []);
  let type, dispose, cursor = 0;
  const hooks = [], cleanups = [];
  const react = {
    useState(initial) {
      const index = cursor++;
      if (!(index in hooks)) hooks[index] = initial;
      return [hooks[index], (value) => { hooks[index] = value; }];
    },
    useRef() { const index = cursor++; return hooks[index] ??= {}; },
    useId: () => "connection",
    useEffect(effect) { const index = cursor++; if (!(index in hooks)) { hooks[index] = true; cleanups.push(effect()); } },
    createElement: (type, props, ...children) => ({ type, props: props ?? {}, children: children.flat() }),
  };
  Object.defineProperty(globalThis, "sessionStorage", { configurable: true, value: {
    getItem: (key) => storage.get(key) ?? null,
    removeItem: (key) => storage.delete(key),
  } });
  Object.defineProperty(globalThis, "localStorage", { configurable: true, value: {
    getItem: (key) => persistent.get(key) ?? null,
    setItem: (key, value) => {
      if (storageFailure) throw new Error("Storage unavailable");
      persistent.set(key, value);
    },
  } });
  globalThis.WebSocket = class {
    constructor(url, requestedProtocol) {
      this.url = url; this.requestedProtocol = requestedProtocol; this.protocol = protocol;
      this.readyState = 0; this.bufferedAmount = 0; this.closed = false; this.frames = [];
      sockets.push(this);
      queueMicrotask(() => { if (!this.closed) { this.readyState = 1; this.onopen?.(); } });
    }
    close(code = 1000) {
      if (this.closed) return;
      this.closed = true; this.readyState = 3;
      this.onclose?.({ code });
    }
    emit(message) { this.onmessage?.({ data: JSON.stringify(message) }); }
    send(data) {
      const request = JSON.parse(data);
      this.frames.push(request);
      if (request.type === "authenticate") {
        if (authenticated === "wait") return;
        if (authenticated && request.token === settings.token) this.emit({ type: "authenticated" });
        else this.close(1008);
        return;
      }
      if (respond) { respond(this, request); return; }
      const result = request.method === "provision" ? { pubkey: bot, ...(request.params.picture ? { picture: request.params.picture } : {}) }
        : request.method === "interrupt" ? { interrupted: true }
        : request.method === "status" ? { pubkey: bot, communities: [{ community, name: "Grok", active: true, connected: true, error: null }] }
        : {};
      this.emit({ jsonrpc: "2.0", id: request.id, result });
    }
  };
  // No desktop runner, agent snapshot, signer, relay publisher, or subscriptions exist here.
  apply({ agents: { register(value) { type = value; } }, react, effect(factory) { dispose = factory(); } });
  function render() { cursor = 0; return type.Create({}); }
  function nodes(tree) {
    if (!tree || typeof tree !== "object") return [];
    return [tree, ...tree.children.flatMap(nodes)];
  }
  async function connect(endpoint = settings.endpoint, token = settings.token) {
    const fields = nodes(render()).filter((node) => node.type === "input");
    fields.find((node) => node.props.type === "url").props.onChange({ target: { value: endpoint } });
    fields.find((node) => node.props.type === "password").props.onChange({ target: { value: token } });
    await nodes(render()).find((node) => node.type === "button").props.onClick();
    return nodes(render()).find((node) => ["status", "alert"].includes(node.props.role));
  }
  t.after(() => {
    dispose(); cleanups.forEach((cleanup) => cleanup?.()); globalThis.WebSocket = OriginalWebSocket;
    if (OriginalStorage) Object.defineProperty(globalThis, "sessionStorage", OriginalStorage);
    else delete globalThis.sessionStorage;
    if (OriginalLocalStorage) Object.defineProperty(globalThis, "localStorage", OriginalLocalStorage);
    else delete globalThis.localStorage;
  });
  return { type, sockets, dispose, persistent, storage, connect, render, nodes };
}

test("credentials require TLS except loopback and never go in URL components", () => {
  for (const endpoint of ["ws://remote.example/buzz/ws", "https://remote.example/buzz/ws", "wss://user:secret@remote.example/buzz/ws", "wss://remote.example/buzz/ws?token=secret", "wss://remote.example/buzz/ws#secret"]) {
    assert.throws(() => remoteSettings(endpoint, "secret"));
  }
  assert.equal(remoteSettings("ws://127.0.0.1/buzz/ws", " token ").token, "token");
  assert.throws(() => remoteSettings(settings.endpoint, "  "), /token/);
  assert.throws(() => remoteSettings(settings.endpoint, "x".repeat(2049)), /token/);
});

test("registers hosted lifecycle without any desktop runtime and authenticates every operation", async (t) => {
  const h = harness(t);
  assert.equal(h.type.id, "missbot");
  for (const key of ["run", "runOnce", "models", "timeoutMs", "retire"]) assert.equal(key in h.type, false);
  assert.deepEqual(h.type.defaults(), { config: {} });
  const creation = { ...input(), picture: "https://images.example/grok.png", config: { token: "must-not-be-forwarded" } };
  assert.deepEqual(await h.type.remote.provision(creation), { pubkey: bot, picture: creation.picture });
  assert.deepEqual(await h.type.remote.provision(creation), { pubkey: bot, picture: creation.picture });
  const auth = ["auth", "owner", "signature", "1793481600"];
  assert.equal(await h.type.remote.activate({ ...creation, pubkey: bot, auth }), undefined);
  assert.equal(await h.type.remote.profile({ ...creation, pubkey: bot }), undefined);
  assert.equal(await h.type.remote.interrupt({ ...input(), pubkey: bot, channelId: "channel", threadRootId: "root" }), true);
  assert.equal(await h.type.remote.remove({ ...input(), pubkey: bot }), undefined);
  const operations = h.sockets.map((socket) => {
    assert.equal(socket.url, settings.endpoint);
    assert.equal(socket.requestedProtocol, "missbot.buzz.v1");
    assert.deepEqual(socket.frames[0], { type: "authenticate", token: settings.token });
    assert.equal(socket.frames.length, 2);
    assert.equal(socket.closed, true);
    assert.equal(JSON.stringify(socket.frames[1]).includes(settings.token), false);
    assert.equal("signal" in socket.frames[1].params, false);
    assert.equal("config" in socket.frames[1].params, false);
    return [socket.frames[1].method, socket.frames[1].params];
  });
  assert.deepEqual(operations, [
    ["provision", { name: "Grok", picture: creation.picture, community }],
    ["provision", { name: "Grok", picture: creation.picture, community }],
    ["activate", { pubkey: bot, name: "Grok", community, auth }],
    ["profile", { pubkey: bot, community, name: "Grok", picture: creation.picture }],
    ["interrupt", { pubkey: bot, community, channelId: "channel", threadRootId: "root" }],
    ["remove", { pubkey: bot, community }],
  ]);
});

test("management operations wait for authentication and reject bad tokens and old protocols", async (t) => {
  const waiting = harness(t, { authenticated: "wait" });
  const pending = waiting.type.remote.provision(input());
  await Promise.resolve();
  assert.equal(waiting.sockets[0].frames.length, 1);
  waiting.sockets[0].emit({ type: "authenticated" });
  assert.deepEqual(await pending, { pubkey: bot });
  for (const options of [{ authenticated: false }, { protocol: "missbot.acp.v1" }]) {
    const h = harness(t, options);
    await assert.rejects(h.type.remote.provision(input()), /authentication failed|protocol is unavailable/);
    assert.equal(h.sockets[0].frames.some((frame) => frame.method), false);
    assert.equal(h.sockets[0].closed, true);
  }
});

test("Connect checks status only and saves credentials locally rather than in agent config", async (t) => {
  const h = harness(t, { configured: false });
  const message = await h.connect();
  assert.equal(message.props.role, "status");
  assert.deepEqual(h.sockets[0].frames[1], { jsonrpc: "2.0", id: 1, method: "status", params: {} });
  assert.deepEqual(JSON.parse(h.persistent.get(STORAGE_KEY)), settings);
  assert.deepEqual(h.type.defaults(), { config: {} });
});

test("Connect reports community failures and never saves failed or invalid connections", async (t) => {
  const status = harness(t, { configured: false, respond(socket, request) {
    socket.emit({ jsonrpc: "2.0", id: request.id, result: { pubkey: bot, communities: [
      { community, name: "Grok", active: true, connected: false, error: "Relay unavailable" },
    ] } });
  } });
  assert.match((await status.connect()).children.join(""), /0 connected.*Relay unavailable/);
  for (const options of [
    { authenticated: false }, { storageFailure: true },
    { respond(socket, request) { socket.emit({ jsonrpc: "2.0", id: request.id, result: {} }); } },
  ]) {
    const h = harness(t, { ...options, configured: false });
    assert.equal((await h.connect()).props.role, "alert");
    assert.equal(h.persistent.size, 0);
    await assert.rejects(h.type.remote.provision(input()), /endpoint and management token/);
  }
});

test("hosted session settings migrate and survive plugin replacement with a fresh session", async (t) => {
  const original = harness(t);
  assert.equal(original.storage.has(STORAGE_KEY), false);
  original.dispose();
  const replacement = harness(t, { configured: false, persistent: original.persistent });
  await replacement.type.remote.provision(input());
  assert.equal(replacement.sockets[0].url, settings.endpoint);
  assert.equal(replacement.sockets[0].frames[0].token, settings.token);
});

test("durable settings win over stale sessions and failed migration retains the session", async (t) => {
  const persistent = new Map([[STORAGE_KEY, JSON.stringify({ ...settings, endpoint: "wss://new.example/buzz/ws" })]]);
  const current = harness(t, { persistent });
  await current.type.remote.provision(input());
  assert.equal(current.sockets[0].url, "wss://new.example/buzz/ws");
  const failed = harness(t, { storageFailure: true });
  await assert.rejects(failed.type.remote.provision(input()), /endpoint and management token/);
  assert.equal(failed.storage.has(STORAGE_KEY), true);
  assert.equal(failed.persistent.size, 0);
});

test("ACP credentials are never migrated into hosted settings or shared config", async (t) => {
  const legacy = JSON.stringify({ endpoint: "wss://old.example/acp/ws", token: "old-acp-secret" });
  const h = harness(t, { configured: false, persistent: new Map([["missbot.remote.v1", legacy]]), session: new Map([["missbot.remote.v1", legacy]]) });
  await assert.rejects(h.type.remote.provision({ ...input(), config: { endpoint: settings.endpoint, token: settings.token } }), /endpoint and management token/);
  assert.equal(h.sockets.length, 0);
  assert.equal(h.persistent.has(STORAGE_KEY), false);
  assert.equal(h.storage.get("missbot.remote.v1"), legacy);
  const fields = h.nodes(h.render()).filter((node) => node.type === "input");
  assert.equal(fields.find((node) => node.props.type === "url").props.value, "wss://missbot-buzz.taile6e57.ts.net/buzz/ws");
  assert.equal(fields.find((node) => node.props.type === "password").props.value, "");
});

test("cancellation and plugin unload close authentication and in-flight operation sockets", async (t) => {
  for (const phase of ["authentication", "operation"]) {
    for (const action of ["abort", "unload"]) {
      const h = harness(t, { authenticated: phase === "authentication" ? "wait" : true, respond() {} });
      const controller = new AbortController();
      const pending = h.type.remote.provision({ ...input(), signal: controller.signal });
      const rejection = assert.rejects(pending, /cancelled|stopped/);
      await Promise.resolve();
      assert.equal(h.sockets[0].frames.length, phase === "authentication" ? 1 : 2);
      if (action === "abort") controller.abort(new Error("cancelled")); else h.dispose();
      await rejection;
      assert.equal(h.sockets[0].closed, true);
    }
  }
  const stopped = harness(t);
  stopped.dispose();
  await assert.rejects(stopped.type.remote.provision(input()), /stopped/);
  assert.equal(stopped.sockets.length, 0);
  const aborted = harness(t);
  const controller = new AbortController();
  controller.abort(new Error("cancelled"));
  await assert.rejects(aborted.type.remote.provision({ ...input(), signal: controller.signal }), /cancelled/);
  assert.equal(aborted.sockets.length, 0);
});

test("management requests have a deadline and close the socket on timeout", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const h = harness(t, { respond() {} });
  const pending = h.type.remote.provision(input());
  const rejection = assert.rejects(pending, /provision timed out/);
  await Promise.resolve();
  t.mock.timers.tick(60000);
  await rejection;
  assert.equal(h.sockets[0].closed, true);
});

test("server errors, disconnects, invalid frames and malformed results reject without replay", async (t) => {
  for (const respond of [
    (socket, request) => socket.emit({ jsonrpc: "2.0", id: request.id, error: { code: -32000, message: settings.token } }),
    (socket) => socket.close(),
    (socket) => socket.onerror(),
    (socket) => socket.onmessage({ data: "not JSON" }),
    (socket) => socket.onmessage({ data: "x".repeat(1024 * 1024 + 1) }),
    (socket, request) => socket.emit({ jsonrpc: "2.0", id: request.id + 1, result: { pubkey: bot } }),
    (socket, request) => socket.emit({ jsonrpc: "2.0", id: request.id, result: [] }),
    (socket, request) => socket.emit({ jsonrpc: "2.0", id: request.id, result: {} }),
  ]) {
    const h = harness(t, { respond });
    await assert.rejects(h.type.remote.provision(input()), (error) => !error.message.includes(settings.token));
    assert.equal(h.sockets.length, 1);
    assert.equal(h.sockets[0].closed, true);
  }
});

test("interrupt returns false for idle conversations and rejects malformed results", async (t) => {
  const idle = harness(t, { respond(socket, request) { socket.emit({ jsonrpc: "2.0", id: request.id, result: { interrupted: false } }); } });
  const place = { ...input(), pubkey: bot, channelId: "channel", threadRootId: "root" };
  assert.equal(await idle.type.remote.interrupt(place), false);
  const malformed = harness(t, { respond(socket, request) { socket.emit({ jsonrpc: "2.0", id: request.id, result: {} }); } });
  await assert.rejects(malformed.type.remote.interrupt(place), /Invalid hosted Missbot interruption/);
});
