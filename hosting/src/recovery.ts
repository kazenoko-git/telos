import { Hono } from "hono";
import { HTTPException } from "hono/http-exception";
import { deleteCookie } from "hono/cookie";
import { account, body, hash, limitAuth, now, passwordHash, sameOrigin, secret, text, type AppEnv } from "./auth";

import { sendAccountMail } from "./email";

type Recipient = { id: string; email: string; salt: string };

// Only a hash is stored. The URL fragment stays out of HTTP requests and access logs.
export async function sendAction(env: Env, recipient: Recipient, purpose: "verify" | "reset"): Promise<boolean> {
  const token = secret("tls_action_");
  const digest = await hash(token);
  const lifetime = purpose === "verify" ? 86400 : 1800;
  await env.DB.prepare("INSERT INTO account_actions(hash,account_id,purpose,credential_salt,expires_at) VALUES(?,?,?,?,?)")
    .bind(digest, recipient.id, purpose, recipient.salt, now() + lifetime).run();
  const url = `${env.ACCOUNT_ORIGIN}/account#${purpose}=${token}`;
  const title = purpose === "verify" ? "Verify your Telos email" : "Reset your Telos password";
  try {
    await sendAccountMail(env, {
      to: recipient.email, subject: title,
      text: `${title}\n\n${url}\n\nThis single-use link expires in ${purpose === "verify" ? "24 hours" : "30 minutes"}. If you did not request this, ignore this email.`,
      html: `<p>${title}</p><p><a href="${url}">Continue to Telos</a></p><p>This single-use link expires in ${purpose === "verify" ? "24 hours" : "30 minutes"}. If you did not request this, ignore this email.</p>`,
    }, "account-" + digest);
    return true;
  } catch {
    await env.DB.prepare("DELETE FROM account_actions WHERE hash=?").bind(digest).run();
    // Do not log addresses, tokens, or provider errors that may contain message bodies.
    console.error(JSON.stringify({ event: "account_email_delivery_failed", purpose }));
    return false;
  }
}

async function deliveryLimit(env: Env, email: string, purpose: string): Promise<void> {
  const bucket = "email:" + await hash(purpose + ":" + email);
  const row = await env.DB.prepare(
    "INSERT INTO auth_limits(bucket,count,expires_at) VALUES(?,1,?) ON CONFLICT(bucket) DO UPDATE SET count=CASE WHEN expires_at<? THEN 1 ELSE count+1 END,expires_at=CASE WHEN expires_at<? THEN excluded.expires_at ELSE expires_at END RETURNING count",
  ).bind(bucket, now()+3600, now(), now()).first<{ count: number }>();
  if (!row || row.count > 3) throw new HTTPException(429, { message: "Too many email requests. Try again later." });
}

export const recovery = new Hono<AppEnv>();
recovery.post("/verification/request", async c => {
  await limitAuth(c);
  const owner = await account(c);
  if (owner.email_verified_at) return c.json({ ok: true });
  await deliveryLimit(c.env, owner.email, "verify");
  const recipient = await c.env.DB.prepare("SELECT id,email,salt FROM accounts WHERE id=?").bind(owner.id).first<Recipient>();
  if (!recipient || !await sendAction(c.env, recipient, "verify")) throw new HTTPException(503, { message: "Email delivery is unavailable. Please try again later." });
  return c.json({ ok: true });
});

recovery.post("/password/request", async c => {
  sameOrigin(c);
  await limitAuth(c);
  const data = await body(c);
  const email = text(data.email, "email", 254).toLowerCase();
  await deliveryLimit(c.env, email, "reset");
  const recipient = await c.env.DB.prepare("SELECT id,email,salt FROM accounts WHERE email=?").bind(email).first<Recipient>();
  if (recipient) await sendAction(c.env, recipient, "reset");
  // Same response for unknown accounts and delivery failures, with no account disclosure.
  return c.json({ ok: true, message: "If that address is registered, a reset link will be sent." });
});

for (const purpose of ["verify", "reset"] as const) {
  recovery.post(purpose === "verify" ? "/verification/confirm" : "/password/reset", async c => {
    sameOrigin(c);
    await limitAuth(c);
    const data = await body(c);
    const token = text(data.token, "email link", 100);
    if (!/^tls_action_[a-f0-9]{64}$/.test(token)) throw new HTTPException(400, { message: "Invalid or expired email link." });
    const digest = await hash(token);
    const valid = "SELECT a.account_id FROM account_actions a JOIN accounts u ON u.id=a.account_id WHERE a.hash=? AND a.purpose=? AND a.expires_at>? AND a.credential_salt=u.salt";
    let operations;
    if (purpose === "verify") {
      operations = await c.env.DB.batch([
        c.env.DB.prepare(`UPDATE accounts SET email_verified_at=COALESCE(email_verified_at,?) WHERE id IN (${valid})`).bind(now(), digest, purpose, now()),
        c.env.DB.prepare("DELETE FROM account_actions WHERE hash=? AND purpose=?").bind(digest, purpose),
      ]);
    } else {
      if (typeof data.password !== "string" || data.password.length < 12 || data.password.length > 512) {
        throw new HTTPException(400, { message: "Use a password of 12–512 characters." });
      }
      const salt = secret("");
      const password = await passwordHash(data.password, salt);
      // D1 batches are atomic. Delete sessions while the old salt still validates the link.
      operations = await c.env.DB.batch([
        c.env.DB.prepare(`DELETE FROM sessions WHERE account_id IN (${valid})`).bind(digest, purpose, now()),
        c.env.DB.prepare(`UPDATE accounts SET password_hash=?,salt=? WHERE id IN (${valid})`).bind(password, salt, digest, purpose, now()),
        c.env.DB.prepare("DELETE FROM account_actions WHERE hash=? AND purpose=?").bind(digest, purpose),
      ]);
      if (operations[1].meta.changes) deleteCookie(c, "__Host-telos_session", { secure: true, path: "/" });
    }
    if (!operations[purpose === "verify" ? 0 : 1].meta.changes) throw new HTTPException(400, { message: "Invalid or expired email link." });
    return c.json({ ok: true });
  });
}
