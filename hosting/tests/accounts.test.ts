import { SignJWT, generateKeyPair, exportJWK } from "jose";
import { env } from "cloudflare:workers";
import { beforeAll, beforeEach, afterEach, expect, it, vi } from "vitest";
import app from "../src/index";
import oauthSchema from "../migrations/0004_social_signin.sql?raw";
import recoverySchema from "../migrations/0003_account_recovery.sql?raw";
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
  const owner = await response.json<{ token: string; id: string }>();
  await env.DB.prepare("UPDATE accounts SET email_verified_at=? WHERE id=?").bind(now(), owner.id).run();
  return owner;
}
async function createKey(token: string) {
  const response = await call("/api/keys", "POST", { name: "fixture" }, token);
  expect(response.status).toBe(201);
  return response.json<{ id: string; key: string }>();
}
let googlePrivate: CryptoKey;
let googleJwk: Record<string, unknown>;
beforeAll(async () => {
  await env.DB.exec((schema + recoverySchema + oauthSchema).replace(/\n/g, " "));
  const keys = await generateKeyPair("RS256", { extractable: true });
  googlePrivate = keys.privateKey;
  googleJwk = { ...await exportJWK(keys.publicKey), kid: "fixture-google", alg: "RS256", use: "sig" };
});
beforeEach(async () => {
  vi.spyOn(env.EMAIL, "send").mockResolvedValue({ messageId: "fixture-email" });
  await env.DB.batch(["oauth_flows", "oauth_identities", "account_actions", "sessions", "api_keys", "hosts", "accounts", "auth_limits"].map(table => env.DB.prepare("DELETE FROM " + table)));
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
    expect(options?.redirect).toBe("manual");
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

it("routes reserved paths with native model IDs and rejects origin redirects", async () => {
  const owner = await signup();
  const key = await createKey(owner.token);
  await call("/api/hosts", "POST", { slug: "tiny", model: "fixture" }, owner.token);
  await env.DB.prepare("UPDATE hosts SET last_seen=? WHERE slug='tiny'").bind(now()).run();
  const root = "https://tls.wingit.tech";
  const listed = await call("/tiny/v1/models", "GET", undefined, key.key, root);
  expect((await listed.json<{ data: {id: string}[] }>()).data.map(row => row.id)).toEqual(["fixture"]);
  const upstream = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, options) => {
    expect(String(input)).toMatch(/\/v1\/completions$/);
    expect(String(input)).not.toContain("/tiny/");
    expect(options?.redirect).toBe("manual");
    return new Response(null, { status: 302, headers: { Location: "https://untrusted.invalid" } });
  });
  expect((await call("/tiny/v1/completions", "POST", { model: "fixture", prompt: "Hi" }, key.key, root)).status).toBe(502);
  expect(upstream).toHaveBeenCalledTimes(1);
  expect((await call("/other/v1/models", "GET", undefined, key.key, root)).status).toBe(200);
  expect((await call("/tiny/v1/models", "GET", undefined, undefined, root)).status).toBe(401);
});

function mailedToken() {
  const message = vi.mocked(env.EMAIL.send).mock.calls.at(-1)![0];
  if (!("text" in message)) throw new Error("Expected text email");
  return String(message.text).match(/tls_action_[a-f0-9]{64}/)![0];
}

it("verifies disposable email once, gates new resources and preserves existing API keys", async () => {
  const response = await call("/api/signup", "POST", { email: "verify@example.invalid", password: "fixture-password-12345" });
  const owner = await response.json<{ token: string; id: string; verification_sent: boolean }>();
  expect(owner.verification_sent).toBe(true);
  const token = mailedToken();
  expect((await call("/api/keys", "POST", { name: "blocked" }, owner.token)).status).toBe(403);
  expect((await call("/api/hosts", "POST", { slug: "blocked", model: "tiny" }, owner.token)).status).toBe(403);
  const row = await env.DB.prepare("SELECT hash FROM account_actions WHERE account_id=?").bind(owner.id).first<{hash: string}>();
  expect(row!.hash).toBe(await hash(token));
  expect(row!.hash).not.toContain(token);
  expect((await call("/api/verification/confirm", "POST", { token })).status).toBe(200);
  const key = await createKey(owner.token);
  expect((await call("/api/verification/confirm", "POST", { token })).status).toBe(400);
  expect((await call("/v1/models", "GET", undefined, key.key, "https://tls.wingit.tech")).status).toBe(200);
});

it("resets once, revokes sessions and outstanding links, and leaves existing keys working", async () => {
  const owner = await signup();
  const key = await createKey(owner.token);
  await call("/api/password/request", "POST", { email: "owner@example.com" });
  const oldLink = mailedToken();
  await call("/api/password/request", "POST", { email: "owner@example.com" });
  const token = mailedToken();
  const results = await Promise.all([1,2].map(() => call("/api/password/reset", "POST", { token, password: "replacement-password-123" })));
  expect(results.map(row => row.status).sort()).toEqual([200,400]);
  expect((await call("/api/me", "GET", undefined, owner.token)).status).toBe(401);
  expect((await call("/api/login", "POST", { email: "owner@example.com", password: "fixture-password-12345" })).status).toBe(401);
  expect((await call("/api/login", "POST", { email: "owner@example.com", password: "replacement-password-123" })).status).toBe(200);
  expect((await call("/api/password/reset", "POST", { token: oldLink, password: "another-password-123" })).status).toBe(400);
  expect((await call("/v1/models", "GET", undefined, key.key, "https://tls.wingit.tech")).status).toBe(200);
});

it("rejects expired, wrong-purpose and malformed links without consuming valid reset links", async () => {
  const owner = await signup();
  const verification = mailedToken();
  expect((await call("/api/password/reset", "POST", { token: verification, password: "replacement-password-123" })).status).toBe(400);
  await call("/api/password/request", "POST", { email: "owner@example.com" });
  const reset = mailedToken();
  expect((await call("/api/verification/confirm", "POST", { token: reset })).status).toBe(400);
  expect((await call("/api/password/reset", "POST", { token: reset, password: "short" })).status).toBe(400);
  await env.DB.prepare("UPDATE account_actions SET expires_at=? WHERE account_id=?").bind(now()-1, owner.id).run();
  expect((await call("/api/password/reset", "POST", { token: reset, password: "replacement-password-123" })).status).toBe(400);
  expect((await call("/api/verification/confirm", "POST", { token: verification })).status).toBe(400);
  expect((await call("/api/password/reset", "POST", { token: "garbage", password: "replacement-password-123" })).status).toBe(400);
});

it("hides account existence, limits email requests, and removes undeliverable tokens", async () => {
  await signup();
  vi.mocked(env.EMAIL.send).mockRejectedValue(new Error("provider unavailable"));
  const known = await call("/api/password/request", "POST", { email: "owner@example.com" });
  const unknown = await call("/api/password/request", "POST", { email: "missing@example.invalid" });
  expect(await known.json()).toEqual(await unknown.json());
  const tokens = await env.DB.prepare("SELECT COUNT(*) AS count FROM account_actions WHERE purpose='reset'").first<{count: number}>();
  expect(tokens!.count).toBe(0);
  for (let i=0; i<2; i++) await call("/api/password/request", "POST", { email: "owner@example.com" });
  expect((await call("/api/password/request", "POST", { email: "owner@example.com" })).status).toBe(429);
});

it("keeps legacy API keys usable before email verification and after it", async () => {
  const owner = await signup();
  const key = await createKey(owner.token);
  await env.DB.prepare("UPDATE accounts SET email_verified_at=NULL WHERE id=?").bind(owner.id).run();
  const requestModels = () => call("/v1/models", "GET", undefined, key.key, "https://tls.wingit.tech");
  expect((await requestModels()).status).toBe(200);
  expect((await call("/api/keys", "POST", { name: "blocked" }, owner.token)).status).toBe(403);
  expect((await call("/api/verification/request", "POST", {}, owner.token)).status).toBe(200);
  expect((await call("/api/verification/confirm", "POST", { token: mailedToken() })).status).toBe(200);
  expect((await requestModels()).status).toBe(200);
});

it("serves the portal only at /account with isolated assets", async () => {
  const assets = vi.spyOn(env.ASSETS, "fetch").mockImplementation(async input => {
    const request = input instanceof Request ? input : new Request(String(input));
    return new Response(new URL(request.url).pathname);
  });
  expect(await (await call("/account")).text()).toBe("/");
  expect(await (await call("/hosting-assets/app.js")).text()).toBe("/app.js");
  assets.mockClear();
  expect((await call("/")).status).toBe(404);
  expect((await call("/research-paper" )).status).toBe(404);
  expect(assets).not.toHaveBeenCalled();
});

async function startOAuth(kind: string, token?: string, intent = "login") {
  const response = await call(`/api/auth/${kind}/start`, "POST", { intent }, token);
  expect(response.status).toBe(200);
  const { url } = await response.json<{url: string}>();
  const state = new URL(url).searchParams.get("state")!;
  expect(response.headers.get("set-cookie")).toMatch(/HttpOnly/);
  expect(response.headers.get("set-cookie")).toMatch(/Secure/);
  expect(response.headers.get("set-cookie")).toMatch(/SameSite=Lax/);
  const cookie = response.headers.get("set-cookie")!.split(";")[0];
  return { state, cookie, url };
}
function callbackOAuth(kind: string, flow: {state: string; cookie: string}, cookie = flow.cookie, code = "fixture-code") {
  return app.fetch(new Request(`${origin}/api/auth/${kind}/callback?state=${flow.state}&code=${code}`, { headers: { Cookie: cookie } }), env);
}
function sessionFrom(response: Response) {
  return response.headers.get("set-cookie")!.match(/__Host-telos_session=([^;,\s]+)/)![1];
}
function githubMock(email = "social@example.invalid", subject = 1234, verified = true) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input, options) => {
    const url = String(input);
    expect(options?.redirect).toBe("manual");
    if (url === "https://github.com/login/oauth/access_token") {
      const form = new URLSearchParams(String(options?.body));
      expect(form.get("code_verifier")).toMatch(/^[a-f0-9]{64}$/);
      expect(form.get("redirect_uri")).toBe(origin + "/api/auth/github/callback");
      return Response.json({ access_token: "fixture-access-token", token_type: "bearer" });
    }
    expect(options?.headers).toMatchObject({ Authorization: "Bearer fixture-access-token" });
    if (url === "https://api.github.com/user") return Response.json({ id: subject });
    if (url === "https://api.github.com/user/emails?per_page=100") return Response.json([{ email, primary: true, verified }]);
    throw new Error("Unexpected provider URL");
  });
}

it("binds OAuth to browser, provider and expiry and consumes it once", async () => {
  const flow = await startOAuth("github");
  expect(new URL(flow.url).searchParams.get("scope")).toBe("user:email");
  expect(new URL(flow.url).searchParams.get("code_challenge_method")).toBe("S256");
  expect(new URL(flow.url).searchParams.get("code_challenge")).toHaveLength(43);
  expect((await callbackOAuth("github", flow, "")).headers.get("location")).toContain("invalid_state");
  expect((await callbackOAuth("google", flow)).headers.get("location")).toContain("invalid_state");
  githubMock();
  const result = await callbackOAuth("github", flow);
  expect(result.headers.get("location")).toBe("/account#auth_success=signed_in");
  expect((await callbackOAuth("github", flow)).headers.get("location")).toContain("invalid_state");
  const session = sessionFrom(result);
  const owner = await (await call("/api/me", "GET", undefined, session)).json<{email_verified_at: number}>();
  expect(owner.email_verified_at).toBeGreaterThan(0);
  const key = await createKey(session);
  const again = await callbackOAuth("github", await startOAuth("github"));
  const sameOwner = await (await call("/api/me", "GET", undefined, sessionFrom(again))).json<{id: string}>();
  expect((await env.DB.prepare("SELECT account_id FROM api_keys WHERE id=?").bind(key.id).first<{account_id: string}>())!.account_id).toBe(sameOwner.id);
  expect((await call("/v1/models", "GET", undefined, key.key, "https://tls.wingit.tech")).status).toBe(200);
  const expired = await startOAuth("github");
  await env.DB.prepare("UPDATE oauth_flows SET expires_at=?").bind(now()-1).run();
  expect((await callbackOAuth("github", expired)).headers.get("location")).toContain("invalid_state");
});

it("requires explicit fresh authenticated linking and preserves existing API keys", async () => {
  const owner = await signup(); const key = await createKey(owner.token);
  githubMock("owner@example.com");
  expect((await callbackOAuth("github", await startOAuth("github"))).headers.get("location")).toContain("sign_in_to_link");
  expect((await call("/api/auth/github/start", "POST", { intent: "link" })).status).toBe(401);
  expect((await callbackOAuth("github", await startOAuth("github", owner.token, "link"))).headers.get("location")).toContain("auth_success=linked");
  const signedIn = await callbackOAuth("github", await startOAuth("github"));
  expect((await (await call("/api/me", "GET", undefined, sessionFrom(signedIn))).json<{id: string}>()).id).toBe(owner.id);
  expect((await call("/v1/models", "GET", undefined, key.key, "https://tls.wingit.tech")).status).toBe(200);
  await env.DB.prepare("UPDATE sessions SET authenticated_at=0").run();
  expect((await call("/api/auth/google/start", "POST", { intent: "link" }, owner.token)).status).toBe(403);
});

it("rejects linking to another owner and invalidates pending links on password reset", async () => {
  const first = await signup(); const second = await signup("second@example.com");
  githubMock();
  await callbackOAuth("github", await startOAuth("github", first.token, "link"));
  expect((await callbackOAuth("github", await startOAuth("github", second.token, "link"))).headers.get("location")).toContain("already_linked");
  const flow = await startOAuth("github", second.token, "link");
  await env.DB.prepare("DELETE FROM sessions WHERE account_id=?").bind(second.id).run();
  expect((await callbackOAuth("github", flow)).headers.get("location")).toContain("reauthenticate");
});

it("rejects unverified GitHub email and provider redirects without creating an account", async () => {
  githubMock("social@example.invalid",1234,false);
  expect((await callbackOAuth("github", await startOAuth("github"))).headers.get("location")).toContain("provider_failed");
  expect((await env.DB.prepare("SELECT COUNT(*) AS count FROM accounts").first<{count: number}>())!.count).toBe(0);
  vi.mocked(globalThis.fetch).mockResolvedValue(new Response(null, { status: 302, headers: { Location: "https://untrusted.invalid" } }));
  expect((await callbackOAuth("github", await startOAuth("github"))).headers.get("location")).toContain("provider_failed");
});

it("sends fallback mail through Resend and discards tokens on quota failures", async () => {
  const configured = { ...env, EMAIL_PROVIDER: "resend" };
  const send = vi.spyOn(globalThis,"fetch").mockImplementation(async (input,options) => {
    expect(String(input)).toBe("https://api.resend.com/emails");
    expect(options?.redirect).toBe("manual");
    expect(options?.headers).toMatchObject({ Authorization: "Bearer fixture-resend-key" });
    const message = JSON.parse(String(options?.body));
    expect(message.to).toEqual(["resend@example.invalid"]);
    expect(message.text).toContain("/account#verify=tls_action_");
    return Response.json({ id: "fixture-email" });
  });
  const response = await app.fetch(new Request(origin+"/api/signup", { method: "POST", headers: { "Content-Type":"application/json" }, body: JSON.stringify({ email: "resend@example.invalid", password: "fixture-password-12345" }) }), configured);
  expect((await response.json<{verification_sent: boolean}>()).verification_sent).toBe(true);
  expect(env.EMAIL.send).not.toHaveBeenCalled();
  send.mockResolvedValue(new Response("quota exceeded",{status:429}));
  const session = response.headers.get("set-cookie")!.split(";")[0];
  const retry = await app.fetch(new Request(origin+"/api/verification/request",{ method:"POST",headers:{ Cookie:session } }),configured);
  expect(retry.status).toBe(503);
  expect((await env.DB.prepare("SELECT COUNT(*) AS count FROM account_actions").first<{count:number}>())!.count).toBe(1);
});

async function googleMock(flow: {state:string}, changes: Record<string,unknown> = {}) {
  const stored = await env.DB.prepare("SELECT nonce FROM oauth_flows WHERE state_hash=?").bind(await hash(flow.state)).first<{nonce:string}>();
  const jwt = await new SignJWT({ sub: "google-subject-123", email: "fixture@gmail.com", email_verified: true, nonce: stored!.nonce, ...changes })
    .setProtectedHeader({ alg: "RS256", kid: "fixture-google" }).setIssuer(typeof changes.iss === "string" ? changes.iss : "https://accounts.google.com")
    .setAudience(typeof changes.aud === "string" ? changes.aud : "fixture-google-client").setIssuedAt().setExpirationTime(typeof changes.exp === "number" ? changes.exp : "5m").sign(googlePrivate);
  vi.spyOn(globalThis,"fetch").mockImplementation(async (input,options) => {
    if (String(input) === "https://www.googleapis.com/oauth2/v3/certs") return Response.json({ keys: [googleJwk] });
    expect(String(input)).toBe("https://oauth2.googleapis.com/token");
    expect(options?.redirect).toBe("manual");
    expect(new URLSearchParams(String(options?.body)).get("code_verifier")).toMatch(/^[a-f0-9]{64}$/);
    return Response.json({ id_token: jwt });
  });
}

it("validates signed Google identity and skips email only for authoritative addresses", async () => {
  const flow = await startOAuth("google");
  await googleMock(flow);
  const response = await callbackOAuth("google",flow);
  expect(response.headers.get("location")).toContain("auth_success=signed_in");
  const owner = await (await call("/api/me","GET",undefined,sessionFrom(response))).json<{email_verified_at:number}>();
  expect(owner.email_verified_at).toBeGreaterThan(0);
  expect(env.EMAIL.send).not.toHaveBeenCalled();
  vi.restoreAllMocks();
  vi.spyOn(env.EMAIL,"send").mockResolvedValue({messageId:"fixture-email"});
  const external = await startOAuth("google");
  await googleMock(external,{sub:"external-subject",email:"external@example.invalid"});
  const result = await callbackOAuth("google",external);
  expect(result.headers.get("location")).toContain("auth_success=signed_in");
  const externalOwner = await (await call("/api/me","GET",undefined,sessionFrom(result))).json<{email_verified_at:number|null}>();
  expect(externalOwner.email_verified_at).toBeNull();
  expect(env.EMAIL.send).toHaveBeenCalledOnce();
});

it.each([{nonce:"wrong-nonce"},{email_verified:false},{azp:"wrong-client"},{iss:"https://untrusted.invalid"},{aud:"wrong-client"},{exp:1}])("rejects invalid Google claims %j",async changes => {
  const flow = await startOAuth("google");
  await googleMock(flow,changes);
  expect((await callbackOAuth("google",flow)).headers.get("location")).toContain("provider_failed");
  expect((await env.DB.prepare("SELECT COUNT(*) AS count FROM accounts").first<{count:number}>())!.count).toBe(0);
});

it("does not expose unconfigured providers or start them",async () => {
  const configured = { ...env, GOOGLE_CLIENT_ID: "", GITHUB_CLIENT_SECRET: "" };
  const available = await app.fetch(new Request(origin+"/api/auth/providers"),configured);
  expect(await available.json()).toEqual({google:false,github:false});
  expect((await app.fetch(new Request(origin+"/api/auth/google/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({intent:"login"})}),configured)).status).toBe(503);
});
