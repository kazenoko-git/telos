import { Hono } from "hono";
import { HTTPException } from "hono/http-exception";
import { deleteCookie, getCookie, setCookie } from "hono/cookie";
import { createRemoteJWKSet, jwtVerify } from "jose";
import { account, body, hash, issueSession, limitAuth, now, sameOrigin, secret, type AppEnv } from "./auth";
import { sendAction } from "./recovery";

type Provider = "google" | "github";
type Identity = { subject: string; email: string; verified: boolean };
type Flow = { provider: Provider; verifier: string; nonce: string; account_id: string | null; session_hash: string | null; credential_salt: string | null };
const googleKeys = createRemoteJWKSet(new URL("https://www.googleapis.com/oauth2/v3/certs"));
const flowCookie = "__Host-telos_oauth";
const cookieOptions = { httpOnly: true, secure: true, sameSite: "Lax" as const, path: "/", maxAge: 600 };
function provider(value: string): Provider {
  if (value !== "google" && value !== "github") throw new HTTPException(404, { message: "Unknown sign-in provider." });
  return value;
}
function config(env: Env, kind: Provider) {
  const id = kind === "google" ? env.GOOGLE_CLIENT_ID : env.GITHUB_CLIENT_ID;
  const key = kind === "google" ? env.GOOGLE_CLIENT_SECRET : env.GITHUB_CLIENT_SECRET;
  if (!id || !key) throw new HTTPException(503, { message: "This sign-in provider is not configured yet." });
  return { id, key, callback: `${env.ACCOUNT_ORIGIN}/api/auth/${kind}/callback` };
}
async function jsonResponse<T>(url: string, options: RequestInit): Promise<T> {
  const response = await fetch(url, { ...options, redirect: "manual", signal: AbortSignal.timeout(15000) });
  if (!response.ok) { await response.body?.cancel(); throw new Error("Provider request rejected"); }
  const reader = response.body!.getReader();
  let result = "";
  const decoder = new TextDecoder();
  let bytes = 0;
  try {
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      bytes += chunk.value.byteLength;
      if (bytes > 131072) { await reader.cancel(); throw new Error("Provider response too large"); }
      result += decoder.decode(chunk.value, { stream: true });
    }
    result += decoder.decode();
  } finally { reader.releaseLock(); }
  return JSON.parse(result) as T;
}

async function identify(env: Env, kind: Provider, flow: Flow, code: string): Promise<Identity> {
  const { id, key, callback } = config(env, kind);
  const token = await jsonResponse<{ access_token?: string; id_token?: string; token_type?: string; scope?: string }>(
    kind === "google" ? "https://oauth2.googleapis.com/token" : "https://github.com/login/oauth/access_token", {
      method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded", Accept: "application/json" },
      body: new URLSearchParams({ client_id: id, client_secret: key, code, redirect_uri: callback, code_verifier: flow.verifier, grant_type: "authorization_code" }),
    });
  if (kind === "google") {
    if (!token.id_token) throw new Error("Missing ID token");
    const { payload } = await jwtVerify(token.id_token, googleKeys, {
      issuer: ["https://accounts.google.com", "accounts.google.com"], audience: id, algorithms: ["RS256"], requiredClaims: ["sub", "exp", "iat", "nonce", "email"],
    });
    if (payload.nonce !== flow.nonce || payload.email_verified !== true || typeof payload.email !== "string" || !payload.sub ||
        (payload.azp !== undefined && payload.azp !== id) || (Array.isArray(payload.aud) && payload.aud.length > 1 && payload.azp !== id)) throw new Error("Invalid Google identity");
    // Google is authoritative for Gmail/Workspace; external email addresses still need verification.
    return { subject: payload.sub, email: payload.email.toLowerCase(), verified: payload.email.toLowerCase().endsWith("@gmail.com") || (typeof payload.hd === "string" && payload.hd.length > 0) };
  }
  if (!token.access_token || token.token_type?.toLowerCase() !== "bearer") throw new Error("Missing GitHub access token");
  const headers = { Authorization: "Bearer " + token.access_token, Accept: "application/vnd.github+json", "User-Agent": "telos-hosting", "X-GitHub-Api-Version": "2026-03-10" };
  const user = await jsonResponse<{ id?: number }>("https://api.github.com/user", { headers });
  const emails = await jsonResponse<{ email: string; primary: boolean; verified: boolean }[]>("https://api.github.com/user/emails?per_page=100", { headers });
  const email = emails.find(row => row.primary === true && row.verified === true);
  if (!Number.isSafeInteger(user.id) || !user.id || !email || typeof email.email !== "string") throw new Error("A verified GitHub email is required");
  return { subject: String(user.id), email: email.email.toLowerCase(), verified: true };
}

export const oauth = new Hono<AppEnv>();
oauth.get("/providers", c => c.json({ google: Boolean(c.env.GOOGLE_CLIENT_ID && c.env.GOOGLE_CLIENT_SECRET), github: Boolean(c.env.GITHUB_CLIENT_ID && c.env.GITHUB_CLIENT_SECRET) }));
oauth.post("/:provider/start", async c => {
  sameOrigin(c);
  await limitAuth(c);
  const kind = provider(c.req.param("provider"));
  const { id, callback } = config(c.env, kind);
  const data = await body(c);
  if (data.intent !== "login" && data.intent !== "link") throw new HTTPException(400, { message: "Choose login or link." });
  let accountId: string | null = null, sessionHash: string | null = null, salt: string | null = null;
  if (data.intent === "link") {
    const owner = await account(c);
    const session = c.req.header("authorization")?.match(/^Bearer (tls_session_[a-f0-9]{64})$/)?.[1] || getCookie(c, "__Host-telos_session");
    sessionHash = await hash(session!);
    const fresh = await c.env.DB.prepare("SELECT accounts.salt FROM sessions JOIN accounts ON accounts.id=sessions.account_id WHERE sessions.hash=? AND sessions.account_id=? AND sessions.authenticated_at>? AND sessions.expires_at>?")
      .bind(sessionHash, owner.id, now()-600, now()).first<{ salt: string }>();
    if (!fresh) throw new HTTPException(403, { message: "Sign in again before linking a provider." });
    accountId = owner.id; salt = fresh.salt;
  }
  const state = secret(""); const browser = secret(""); const verifier = secret(""); const nonce = secret("");
  // Expired state rows contain short-lived PKCE material; remove them on each new flow.
  await c.env.DB.batch([
    c.env.DB.prepare("DELETE FROM oauth_flows WHERE expires_at<=?").bind(now()),
    c.env.DB.prepare("INSERT INTO oauth_flows(state_hash,browser_hash,provider,verifier,nonce,account_id,session_hash,credential_salt,expires_at) VALUES(?,?,?,?,?,?,?,?,?)")
      .bind(await hash(state), await hash(browser), kind, verifier, nonce, accountId, sessionHash, salt, now()+600),
  ]);
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier));
  const challenge = btoa(String.fromCharCode(...new Uint8Array(digest))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  const url = new URL(kind === "google" ? "https://accounts.google.com/o/oauth2/v2/auth" : "https://github.com/login/oauth/authorize");
  url.search = new URLSearchParams({ client_id: id, redirect_uri: callback, response_type: "code", scope: kind === "google" ? "openid email" : "user:email", state, code_challenge: challenge, code_challenge_method: "S256", ...(kind === "google" ? { nonce, prompt: "select_account" } : {}) }).toString();
  setCookie(c, flowCookie, browser, cookieOptions);
  return c.json({ url: url.toString() });
});

oauth.get("/:provider/callback", async c => {
  const kind = provider(c.req.param("provider"));
  const state = c.req.query("state") || "";
  const browser = getCookie(c, flowCookie) || "";
  if (!/^[a-f0-9]{64}$/.test(state) || !/^[a-f0-9]{64}$/.test(browser)) return c.redirect("/account#auth_error=invalid_state");
  const flow = await c.env.DB.prepare("DELETE FROM oauth_flows WHERE state_hash=? AND browser_hash=? AND provider=? AND expires_at>? RETURNING provider,verifier,nonce,account_id,session_hash,credential_salt")
    .bind(await hash(state), await hash(browser), kind, now()).first<Flow>();
  if (!flow) return c.redirect("/account#auth_error=invalid_state");
  deleteCookie(c, flowCookie, { secure: true, path: "/" });
  const code = c.req.query("code");
  if (c.req.query("error") || !code || code.length > 4096) return c.redirect("/account#auth_error=cancelled");
  try {
    const identity = await identify(c.env, kind, flow, code);
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(identity.email) || identity.email.length > 254) throw new Error("Invalid email");
    const linked = await c.env.DB.prepare("SELECT account_id FROM oauth_identities WHERE provider=? AND subject=?").bind(kind, identity.subject).first<{ account_id: string }>();
    if (flow.account_id) {
      const live = await c.env.DB.prepare("SELECT accounts.id FROM sessions JOIN accounts ON accounts.id=sessions.account_id WHERE sessions.hash=? AND sessions.account_id=? AND sessions.expires_at>? AND accounts.salt=?")
        .bind(flow.session_hash, flow.account_id, now(), flow.credential_salt).first();
      if (!live) return c.redirect("/account#auth_error=reauthenticate");
      if (linked && linked.account_id !== flow.account_id) return c.redirect("/account#auth_error=already_linked");
      if (!linked) {
        const inserted = await c.env.DB.prepare("INSERT INTO oauth_identities(provider,subject,account_id,created_at) SELECT ?,?,?,? WHERE EXISTS(SELECT 1 FROM sessions JOIN accounts ON accounts.id=sessions.account_id WHERE sessions.hash=? AND sessions.account_id=? AND sessions.expires_at>? AND accounts.salt=?)")
          .bind(kind, identity.subject, flow.account_id, now(), flow.session_hash, flow.account_id, now(), flow.credential_salt).run();
        if (!inserted.meta.changes) return c.redirect("/account#auth_error=reauthenticate");
      }
      // Linking establishes a second sign-in method; it never changes the account's email or API keys.
      await issueSession(c, flow.account_id);
      return c.redirect("/account#auth_success=linked");
    }
    if (linked) {
      await issueSession(c, linked.account_id);
      return c.redirect("/account#auth_success=signed_in");
    }
    const collision = await c.env.DB.prepare("SELECT id FROM accounts WHERE email=?").bind(identity.email).first();
    if (collision) return c.redirect("/account#auth_error=sign_in_to_link");
    const accountId = crypto.randomUUID(); const salt = secret("");
    await c.env.DB.batch([
      c.env.DB.prepare("INSERT INTO accounts(id,email,password_hash,salt,created_at,email_verified_at) VALUES(?,?,?,?,?,?)")
        .bind(accountId, identity.email, "", salt, now(), identity.verified ? now() : null),
      c.env.DB.prepare("INSERT INTO oauth_identities(provider,subject,account_id,created_at) VALUES(?,?,?,?)")
        .bind(kind, identity.subject, accountId, now()),
    ]);
    await issueSession(c, accountId);
    if (!identity.verified) await sendAction(c.env, { id: accountId, email: identity.email, salt }, "verify");
    return c.redirect("/account#auth_success=signed_in");
  } catch {
    // Provider responses, authorization codes, ID tokens and access tokens stay out of logs.
    return c.redirect("/account#auth_error=provider_failed");
  }
});
