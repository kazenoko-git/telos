import type { Context } from "hono";
import { HTTPException } from "hono/http-exception";
import { getCookie } from "hono/cookie";

export type Account = { id: string; email: string; email_verified_at: number | null };
export type AppEnv = { Bindings: Env; Variables: { account: Account } };
export const now = () => Math.floor(Date.now() / 1000);

export function secret(prefix: string): string {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  return prefix + Array.from(bytes, b => b.toString(16).padStart(2, "0")).join("");
}

export async function hash(value: string): Promise<string> {
  const bytes = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return Array.from(new Uint8Array(bytes), b => b.toString(16).padStart(2, "0")).join("");
}

export async function passwordHash(password: string, salt: string): Promise<string> {
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(password), "PBKDF2", false, ["deriveBits"]);
  const bytes = await crypto.subtle.deriveBits(
    { name: "PBKDF2", hash: "SHA-256", salt: new TextEncoder().encode(salt), iterations: 100000 }, key, 256,
  );
  return Array.from(new Uint8Array(bytes), b => b.toString(16).padStart(2, "0")).join("");
}

export async function body(c: Context<AppEnv>): Promise<Record<string, unknown>> {
  if (!c.req.header("content-type")?.startsWith("application/json")) {
    throw new HTTPException(415, { message: "Use application/json." });
  }
  const reader = c.req.raw.body?.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  if (!reader) throw new HTTPException(400, { message: "JSON body required." });
  try {
    while (true) {
      const part = await reader.read();
      if (part.done) break;
      size += part.value.byteLength;
      if (size > 4 * 1024 * 1024) {
        await reader.cancel();
        throw new HTTPException(413, { message: "Request body exceeds 4 MiB." });
      }
      chunks.push(part.value);
    }
  } finally {
    reader.releaseLock();
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  try {
    const parsed: unknown = JSON.parse(new TextDecoder().decode(bytes));
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed as Record<string, unknown>;
  } catch {}
  throw new HTTPException(400, { message: "JSON object required." });
}

export function text(value: unknown, label: string, max = 200): string {
  if (typeof value !== "string" || !value.trim() || value.length > max) {
    throw new HTTPException(400, { message: `Invalid ${label}.` });
  }
  return value.trim();
}

export async function limitAuth(c: Context<AppEnv>): Promise<void> {
  const bucket = await hash(c.req.header("cf-connecting-ip") || "local");
  const row = await c.env.DB.prepare(
    "INSERT INTO auth_limits(bucket,count,expires_at) VALUES(?,1,?) ON CONFLICT(bucket) DO UPDATE SET count=CASE WHEN expires_at<? THEN 1 ELSE count+1 END, expires_at=CASE WHEN expires_at<? THEN excluded.expires_at ELSE expires_at END RETURNING count",
  ).bind(bucket, now() + 900, now(), now()).first<{ count: number }>();
  if (!row || row.count > 30) throw new HTTPException(429, { message: "Too many authentication attempts. Try again later." });
}

export function sameOrigin(c: Context<AppEnv>): void {
  const origin = c.req.header("origin");
  if (origin && origin !== c.env.ACCOUNT_ORIGIN) throw new HTTPException(403, { message: "Cross-origin account request rejected." });
}

export async function account(c: Context<AppEnv>): Promise<Account> {
  sameOrigin(c);
  const bearer = c.req.header("authorization")?.match(/^Bearer (tls_session_[a-f0-9]{64})$/)?.[1];
  const token = bearer || getCookie(c, "__Host-telos_session");
  if (!token) throw new HTTPException(401, { message: "Sign in to your Telos account." });
  const found = await c.env.DB.prepare(
    "SELECT accounts.id,accounts.email,accounts.email_verified_at FROM sessions JOIN accounts ON accounts.id=sessions.account_id WHERE sessions.hash=? AND sessions.expires_at>?",
  ).bind(await hash(token), now()).first<Account>();
  if (!found) throw new HTTPException(401, { message: "Session expired or revoked." });
  return found;
}

export async function inferenceAccount(c: Context<AppEnv>): Promise<string> {
  const token = c.req.header("authorization")?.match(/^Bearer (tls_key_[a-f0-9]{64})$/)?.[1];
  const found = token && await c.env.DB.prepare(
    "SELECT account_id FROM api_keys WHERE hash=? AND revoked_at IS NULL",
  ).bind(await hash(token)).first<{ account_id: string }>();
  if (!found) throw new HTTPException(401, { message: "Invalid or revoked API key." });
  return found.account_id;
}

export function requireVerified(c: Context<AppEnv>): void {
  if (!c.get("account").email_verified_at) throw new HTTPException(403, { message: "Verify your email before creating keys or reserving endpoints." });
}
