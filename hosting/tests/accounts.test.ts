import { env } from "cloudflare:workers";
import { beforeAll, beforeEach, afterEach, expect, it, vi } from "vitest";
import app from "../src/index";
import schema from "../migrations/0001_accounts.sql?raw";
import { hash, now } from "../src/auth";

declare module "cloudflare:workers" { interface ProvidedEnv extends Env {} }

const origin = "https://telos.wingit.tech";
function call(path: string, method = "GET", data?: unknown, token?: string, hostname = origin) {
  return app.fetch(new Request(hostname + path, {
    method,
    headers: { "Content-Type": "application/json", ...(token ? { Authorization: "Bearer " + token } : {}) },
    body: data === undefined ? undefined : JSON.stringify(data),
  }), env);
}
async function signup(email = "owner@example.com") {
  const response = await call("/api/signup", "POST", { email, password: "fixture-password-12345" });
  expect(response.status).toBe(200);
  return response.json<{ token: string; id: string }>();
}
async function createKey(token: string) {
  const response = await call("/api/keys", "POST", { name: "fixture" }, token);
  expect(response.status).toBe(201);
  return response.json<{ id: string; key: string }>();
}
beforeAll(async () => { await env.DB.exec(schema.replace(/\n/g, " ")); });
beforeEach(async () => {
  await env.DB.batch(["sessions", "api_keys", "hosts", "accounts", "auth_limits"].map(table => env.DB.prepare("DELETE FROM " + table)));
});
afterEach(() => vi.restoreAllMocks());

it("persists keys across sessions and revokes only the owning account's keys", async () => {
  const owner = await signup();
  const key = await createKey(owner.token);
  const stranger = await signup("stranger@example.com");
  expect((await call("/api/keys/" + key.id, "DELETE", undefined, stranger.token)).status).toBe(404);
  expect((await call("/v1/models", "GET", undefined, key.key, "https://tls.wingit.tech")).status).toBe(200);
  const login = await call("/api/login", "POST", { email: "owner@example.com", password: "fixture-password-12345" });
  const next = await login.json<{ token: string }>();
  await call("/api/logout", "POST", {}, owner.token);
  expect((await call("/api/me", "GET", undefined, owner.token)).status).toBe(401);
  expect((await call("/api/me", "GET", undefined, next.token)).status).toBe(200);
  expect((await call("/v1/models", "GET", undefined, key.key, "https://tls.wingit.tech")).status).toBe(200);
  expect((await call("/api/keys/" + key.id, "DELETE", undefined, next.token)).status).toBe(200);
  expect((await call("/v1/models", "GET", undefined, key.key, "https://tls.wingit.tech")).status).toBe(401);
  const row = await env.DB.prepare("SELECT hash FROM api_keys WHERE id=?").bind(key.id).first<{ hash: string }>();
  expect(row?.hash).toBe(await hash(key.key));
  expect(row?.hash).not.toBe(key.key);
  const listing = await (await call("/api/keys", "GET", undefined, next.token)).text();
  expect(listing).not.toContain(key.key);
});

it("allows exactly one concurrent rotation and immediately rejects the old key", async () => {
  const owner = await signup();
  const old = await createKey(owner.token);
  const results = await Promise.all([1, 2].map(() => call("/api/keys/" + old.id + "/rotate", "POST", {}, owner.token)));
  expect(results.filter(r => r.status === 200)).toHaveLength(1);
  const next = await results.find(r => r.status === 200)!.json<{ key: string }>();
  expect((await call("/v1/models", "GET", undefined, old.key, "https://tls.wingit.tech")).status).toBe(401);
  expect((await call("/v1/models", "GET", undefined, next.key, "https://tls.wingit.tech")).status).toBe(200);
});

it("reserves names atomically and isolates private hosts from the shared pool", async () => {
  const owner = await signup();
  const other = await signup("other@example.com");
  const key = await createKey(other.token);
  const attempts = await Promise.all([owner, other].map(a => call("/api/hosts", "POST", { slug: "tiny", model: "fixture" }, a.token)));
  expect(attempts.map(a => a.status).sort()).toEqual([201, 409]);
  const host = await env.DB.prepare("SELECT account_id FROM hosts WHERE slug='tiny'").first<{ account_id: string }>();
  const winner = host!.account_id === owner.id ? owner : other;
  const loser = host!.account_id === owner.id ? other : owner;
  const loserKey = winner.id === other.id ? await createKey(owner.token) : key;
  await env.DB.prepare("UPDATE hosts SET last_seen=? WHERE slug='tiny'").bind(now()).run();
  const models = async () => (await call("/v1/models", "GET", undefined, loserKey.key, "https://tls.wingit.tech")).json<{ data: unknown[] }>();
  expect((await models()).data).toHaveLength(0);
  expect((await call("/api/hosts/tiny", "PATCH", { shared: true }, loser.token)).status).toBe(404);
  await call("/api/hosts/tiny", "PATCH", { shared: true }, winner.token);
  expect((await models()).data).toHaveLength(1);
  await call("/api/hosts/tiny", "PATCH", { shared: false }, winner.token);
  expect((await models()).data).toHaveLength(0);
});

it("streams without buffering, rewrites the pool model, strips client keys, and propagates cancellation", async () => {
  const owner = await signup();
  const key = await createKey(owner.token);
  await call("/api/hosts", "POST", { slug: "tiny", model: "fixture" }, owner.token);
  await env.DB.prepare("UPDATE hosts SET last_seen=?,shared=1 WHERE slug='tiny'").bind(now()).run();
  const host = await env.DB.prepare("SELECT origin_key FROM hosts WHERE slug='tiny'").first<{ origin_key: string }>();
  let cancelled = false;
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, options) => {
    expect(String(input)).toContain("/v1/chat/completions");
    expect(options?.headers).toMatchObject({ Authorization: "Bearer " + host!.origin_key });
    expect(options?.headers).not.toMatchObject({ Authorization: "Bearer " + key.key });
    expect(JSON.parse(String(options?.body)).model).toBe("fixture");
    return new Response(new ReadableStream({
      start(controller) { controller.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"tiny"}}]}\n\n')); },
      cancel() { cancelled = true; },
    }), { headers: { "Content-Type": "text/event-stream" } });
  });
  const response = await call("/v1/chat/completions", "POST", {
    model: "tiny/fixture", stream: true, messages: [{ role: "user", content: "Hi" }],
  }, key.key, "https://tls.wingit.tech");
  expect(response.status).toBe(200);
  const reader = response.body!.getReader();
  expect(new TextDecoder().decode((await reader.read()).value)).toContain("tiny");
  await reader.cancel();
  expect(cancelled).toBe(true);
});

it("rejects expired sessions, invalid credentials and cross-origin management", async () => {
  const owner = await signup();
  expect((await call("/api/login", "POST", { email: "owner@example.com", password: "wrong-password-123" })).status).toBe(401);
  await env.DB.prepare("UPDATE sessions SET expires_at=?").bind(now()-1).run();
  expect((await call("/api/me", "GET", undefined, owner.token)).status).toBe(401);
  expect((await call("/api/me", "GET", undefined, owner.token, "https://tiny.tls.wingit.tech")).status).toBe(404);
});

it("leases a provisioned connection and rejects stale disconnects", async () => {
  const owner = await signup();
  await call("/api/hosts", "POST", { slug: "tiny", model: "fixture" }, owner.token);
  const requests: string[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, options) => {
    const url = String(input);
    requests.push((options?.method || "GET") + " " + url);
    let result: unknown = {};
    if (url.includes("cfd_tunnel?")) result = [];
    else if (url.endsWith("/cfd_tunnel")) result = { id: "fixture-tunnel" };
    else if (url.endsWith("/token")) result = "fixture-connector-token";
    else if (url.includes("/dns_records?")) result = [];
    return Response.json({ success: true, result });
  });
  const connected = await call("/api/hosts/tiny/connect", "POST", { model: "fixture", port: 8000 }, owner.token);
  expect(connected.status).toBe(200);
  const config = await connected.json<{ connection_id: string; token: string }>();
  expect(config.token).toBe("fixture-connector-token");
  expect((await call("/api/hosts/tiny/connect", "POST", { model: "fixture", port: 8000 }, owner.token)).status).toBe(409);
  await call("/api/hosts/tiny/disconnect", "POST", { connection_id: "stale" }, owner.token);
  const retained = await env.DB.prepare("SELECT connection_id FROM hosts WHERE slug='tiny'").first<{ connection_id: string }>();
  expect(retained?.connection_id).toBe(config.connection_id);
  await call("/api/hosts/tiny/disconnect", "POST", { connection_id: config.connection_id }, owner.token);
  expect((await call("/api/hosts/tiny/connect", "POST", { model: "fixture", port: 8001 }, owner.token)).status).toBe(200);
  expect(requests.filter(r => r.startsWith("POST") && r.endsWith("/cfd_tunnel"))).toHaveLength(1);
});

it("recovers partial provisioning without creating another named tunnel", async () => {
  const owner = await signup();
  await call("/api/hosts", "POST", { slug: "tiny", model: "fixture" }, owner.token);
  let createCount = 0;
  let fail = true;
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, options) => {
    const url = String(input);
    if (url.endsWith("/configurations") && fail) return new Response("fixture failure", { status: 502 });
    let result: unknown = {};
    if (url.includes("cfd_tunnel?")) result = [];
    else if (url.endsWith("/cfd_tunnel") && options?.method === "POST") { createCount++; result = { id: "fixture-tunnel" }; }
    else if (url.endsWith("/token")) result = "fixture-token";
    else if (url.includes("/dns_records?")) result = [];
    return Response.json({ success: true, result });
  });
  expect((await call("/api/hosts/tiny/connect", "POST", { model: "fixture", port: 8000 }, owner.token)).status).toBe(502);
  fail = false;
  expect((await call("/api/hosts/tiny/connect", "POST", { model: "fixture", port: 8000 }, owner.token)).status).toBe(200);
  expect(createCount).toBe(1);
});
