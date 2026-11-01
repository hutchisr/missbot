import assert from "node:assert/strict";
import { test } from "node:test";
import { apply, remoteSettings, RemoteConnection } from "./plugin.js";

const OriginalWebSocket = globalThis.WebSocket;
const OriginalStorage = Object.getOwnPropertyDescriptor(globalThis, "sessionStorage");
const author = "a".repeat(64), bot = "b".repeat(64), root = "c".repeat(64), nested = "d".repeat(64);
const eventId = (n) => n.toString(16).padStart(64, "0");

function harness(t, { privateAck = true, answer, publishError, authenticated = true, configured = true } = {}) {
  const sockets = [], posts = [];
  let type, dispose, changed;
  let running = [{ key: "community/agent" }];
  const settings = { endpoint: "wss://missbot.example/acp/ws", token: "private-token" };
  const storage = new Map(configured ? [["missbot.remote.v1", JSON.stringify(settings)]] : []);
  Object.defineProperty(globalThis, "sessionStorage", { configurable: true, value: {
    getItem: (key) => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value),
  } });
  globalThis.WebSocket = class {
    constructor(url, protocol) {
      this.url = url; this.protocol = protocol; this.readyState = 0; this.bufferedAmount = 0;
      this.closed = false; this.prompts = []; this.frames = [];
      this.session = `session-${sockets.length}`;
      sockets.push(this);
      queueMicrotask(() => { if (!this.closed) { this.readyState = 1; this.onopen(); } });
    }
    close(code = 1000) {
      if (this.closed) return;
      this.closed = true; this.readyState = 3;
      this.onclose?.({ code });
    }
    emit(message) { this.onmessage({ data: JSON.stringify(message) }); }
    send(data) {
      const request = JSON.parse(data);
      this.frames.push(request);
      if (request.type === "authenticate") {
        if (authenticated && request.token === settings.token) this.emit({ type: "authenticated" });
        else this.close(1008);
        return;
      }
      if (request.method === "initialize") this.emit({ jsonrpc: "2.0", id: request.id, result: { protocolVersion: 1 } });
      if (request.method === "session/new") this.emit({ jsonrpc: "2.0", id: request.id, result: {
        sessionId: this.session, _meta: { missbot: { private: privateAck } },
      } });
      if (request.method === "session/prompt") {
        this.prompts.push(request.params.prompt[0].text);
        if (answer === "wait") return;
        const response = answer ?? { text: `turn ${this.prompts.length}`, stop: "end_turn" };
        this.emit({ jsonrpc: "2.0", method: "session/update", params: { sessionId: this.session, update: {
          sessionUpdate: "agent_message_chunk", content: { type: "text", text: response.text },
        } } });
        this.emit({ jsonrpc: "2.0", id: request.id, result: { stopReason: response.stop } });
      }
    }
  };
  apply({
    agents: {
      register(value) { type = value; }, subscribe(callback) { changed = callback; return () => {}; },
      snapshot() { return { status: "ready", running }; },
    },
    effect(factory) { dispose = factory(); },
  });
  t.after(() => {
    dispose(); globalThis.WebSocket = OriginalWebSocket;
    if (OriginalStorage) Object.defineProperty(globalThis, "sessionStorage", OriginalStorage);
    else delete globalThis.sessionStorage;
  });
  function delivery(n, { tags = [], channel = "channel", kind = "stream", solo = false, key = "community/agent", content = "hello" } = {}) {
    return {
      config: {}, channelId: channel, signal: new AbortController().signal,
      trigger: { type: "mention", event: {
        id: eventId(n), pubkey: author, kind: 9, created_at: 1793481600, content, tags: [["h", channel], ...tags],
      } },
      agent: {
        key, pubkey: bot,
        async query() { return [{ kind: 39000, tags: [["d", channel], ["t", kind], ...(solo ? [["private"], ["t", "buzz.solo"]] : [])] }]; },
        async publish(event, check) {
          check(); posts.push(event);
          if (publishError) throw new Error("relay acknowledgement lost");
          return { ...event, id: eventId(999) };
        },
      },
    };
  }
  return { type, sockets, posts, delivery, dispose, settings, stopAgents() { running = []; changed(); } };
}

test("credentials require TLS except loopback and never go in URL components", () => {
  for (const endpoint of ["ws://remote.example/acp/ws", "https://remote.example/acp/ws", "wss://user:secret@remote.example/acp/ws", "wss://remote.example/acp/ws?token=secret", "wss://remote.example/acp/ws#secret"]) {
    assert.throws(() => remoteSettings(endpoint, "secret"));
  }
  assert.equal(remoteSettings("ws://127.0.0.1/acp/ws", " token ").token, "token");
  assert.throws(() => remoteSettings("wss://remote.example/acp/ws", "  "), /token/);
});

test("authentication precedes ACP and failed authentication never sends conversation data", async (t) => {
  const h = harness(t);
  await h.type.run(h.delivery(1));
  assert.equal(h.sockets[0].url, "wss://missbot.example/acp/ws");
  assert.deepEqual(h.sockets[0].frames[0], { type: "authenticate", token: "private-token" });
  assert.equal(h.sockets[0].frames[1].method, "initialize");
  const rejected = harness(t, { authenticated: false });
  await assert.rejects(rejected.type.run(rejected.delivery(2)), /authentication failed/);
  assert.deepEqual(rejected.sockets[0].prompts, []);
  assert.equal(rejected.sockets[0].frames.length, 1);
  const unset = harness(t, { configured: false });
  await assert.rejects(unset.type.run(unset.delivery(3)), /endpoint and access token/);
  assert.equal(unset.sockets.length, 0);
});

test("threads retain context without mixing communities, agents, or channels", async (t) => {
  const h = harness(t);
  await h.type.run(h.delivery(1, { tags: [["e", root, "", "reply"]] }));
  await h.type.run(h.delivery(2, { tags: [["e", root, "", "root"], ["e", nested, "", "reply"]] }));
  await h.type.run(h.delivery(3, { tags: [["e", root, "", "reply"]], key: "other-community/agent" }));
  await h.type.run(h.delivery(4, { tags: [["e", root, "", "reply"]], channel: "other-channel" }));
  await h.type.run(h.delivery(5, { tags: [["e", root, "", "reply"]], key: "community/other-agent" }));
  assert.deepEqual(h.posts.map((post) => post.content), ["turn 1", "turn 2", "turn 1", "turn 1", "turn 1"]);
  assert.deepEqual(h.posts[1].tags, [["h", "channel"], ["e", root, "", "reply"]]);
});

test("DMs and Solo chats reply at channel level; lone root tags do not create threads", async (t) => {
  const h = harness(t);
  await h.type.run(h.delivery(1, { kind: "dm" }));
  await h.type.run(h.delivery(2, { kind: "dm" }));
  await h.type.run(h.delivery(3, { channel: "solo", solo: true }));
  await h.type.run(h.delivery(4, { channel: "solo", solo: true }));
  await h.type.run(h.delivery(5, { tags: [["e", root, "", "root"]] }));
  assert.deepEqual(h.posts.map((post) => post.content), ["turn 1", "turn 2", "turn 1", "turn 2", "turn 1"]);
  assert.deepEqual(h.posts.slice(0, 4).map((post) => post.tags), [
    [["h", "channel"]], [["h", "channel"]], [["h", "solo"]], [["h", "solo"]],
  ]);
  assert.deepEqual(h.posts[4].tags, [["h", "channel"], ["e", eventId(5), "", "reply"]]);
});

test("an old server cannot receive private content; ambiguous destinations fail closed", async (t) => {
  const h = harness(t, { privateAck: false });
  await assert.rejects(h.type.run(h.delivery(1)), /acknowledge private/);
  assert.deepEqual(h.sockets[0].prompts, []);
  assert.equal(h.sockets[0].closed, true);
  const ambiguous = h.delivery(2);
  ambiguous.trigger.event.tags.push(["h", "other"]);
  await assert.rejects(h.type.run(ambiguous), /unambiguous/);
  assert.deepEqual(h.posts, []);
});

test("suppression, refusal and incomplete output never become Buzz messages", async (t) => {
  for (const answer of [
    { text: "NO_REPLY", stop: "end_turn" }, { text: "partial", stop: "cancelled" },
    { text: "partial", stop: "refusal" }, { text: "partial", stop: "max_tokens" },
  ]) {
    const h = harness(t, { answer });
    if (answer.stop === "max_tokens") await assert.rejects(h.type.run(h.delivery(1)), /complete reply/);
    else await h.type.run(h.delivery(1));
    assert.deepEqual(h.posts, []);
  }
});

test("uncertain publication is not replayed and the queue recovers after failure", async (t) => {
  const h = harness(t, { publishError: true });
  await assert.rejects(h.type.run(h.delivery(1)), /acknowledgement lost/);
  await h.type.run(h.delivery(1));
  await assert.rejects(h.type.run(h.delivery(2)), /acknowledgement lost/);
  assert.equal(h.posts.length, 2);
});

test("Stop, agent removal, and plugin unload close in-flight work without publication", async (t) => {
  for (const action of ["interrupt", "remove", "unload"]) {
    const h = harness(t, { answer: "wait" });
    let stop;
    const d = h.delivery(1);
    d.interruptible = (callback) => { stop = callback; };
    const pending = h.type.run(d);
    const rejection = assert.rejects(pending, /cancelled|closed/);
    while (!h.sockets[0]?.prompts.length) await new Promise((resolve) => setImmediate(resolve));
    if (action === "interrupt") await stop();
    if (action === "remove") h.stopAgents();
    if (action === "unload") h.dispose();
    await rejection;
    assert.equal(h.sockets[0].closed, true);
    assert.deepEqual(h.posts, []);
  }
});

test("LRU eviction and protocol errors release connections instead of leaking slots", async (t) => {
  const h = harness(t);
  for (let n = 1; n <= 9; n++) await h.type.run(h.delivery(n));
  assert.equal(h.sockets[0].closed, true);
  assert.equal(h.sockets.filter((socket) => !socket.closed).length, 8);
  const connection = new RemoteConnection(h.settings);
  await connection.open(new AbortController().signal);
  h.sockets.at(-1).onmessage({ data: "not JSON" });
  await assert.rejects(connection.prompt("private"), /closed/);
  h.dispose();
  assert.equal(h.sockets.every((socket) => socket.closed), true);
  await assert.rejects(h.type.run(h.delivery(10)), /stopped/);
});
