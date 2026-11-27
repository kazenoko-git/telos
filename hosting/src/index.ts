import { Hono, type Handler } from "hono";
import { HTTPException } from "hono/http-exception";
import { deleteCookie, getCookie, setCookie } from "hono/cookie";
import { account, body, hash, inferenceAccount, limitAuth, now, passwordHash, sameOrigin, secret, text, requireVerified, type AppEnv } from "./auth";
import { connect, originHost, ownedHost, type Host } from "./tunnels";

import { recovery, sendAction } from "./recovery";

const app = new Hono<AppEnv>();
app.onError((error, c) => {
  if (!(error instanceof HTTPException)) {
    console.error(JSON.stringify({ event: "hosting_request_failed", path: c.req.path, message: error.message }));
  }
  return c.json({
    error: { message: error instanceof HTTPException ? error.message : "The hosting service could not complete this request.", type: "request_error" },
  }, error instanceof HTTPException ? error.status : 500);
});

app.use("*", async (c, next) => {
  if (c.req.path.startsWith("/api/") && new URL(c.req.url).origin !== c.env.ACCOUNT_ORIGIN) {
    throw new HTTPException(404, { message: "Use the Telos account service for management requests." });
  }
  await next();
  c.header("X-Content-Type-Options", "nosniff");
  c.header("Referrer-Policy", "no-referrer");
  c.header("Cache-Control", "no-store");
  c.header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'");
});

for (const action of ["signup", "login"]) {
  app.post("/api/" + action, async c => {
    sameOrigin(c);
    await limitAuth(c);
    const data = await body(c);
    const email = text(data.email, "email", 254).toLowerCase();
    const password = data.password;
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email) || typeof password !== "string" || password.length < 12 || password.length > 512) {
      throw new HTTPException(400, { message: "Use a valid email and a password of 12–512 characters." });
    }
    let owner: { id: string; password_hash: string; salt: string } | null;
    if (action === "signup") {
      const salt = secret("");
      const id = crypto.randomUUID();
      const inserted = await c.env.DB.prepare(
        "INSERT INTO accounts(id,email,password_hash,salt,created_at) VALUES(?,?,?,?,?) ON CONFLICT(email) DO NOTHING",
      ).bind(id, email, await passwordHash(password, salt), salt, now()).run();
      if (!inserted.meta.changes) throw new HTTPException(409, { message: "This email is already registered." });
      owner = { id, password_hash: "", salt };
    } else {
      owner = await c.env.DB.prepare("SELECT id,password_hash,salt FROM accounts WHERE email=?").bind(email)
        .first<{ id: string; password_hash: string; salt: string }>();
      const candidate = await passwordHash(password, owner?.salt || "invalid-account-dummy-salt");
      const a = new TextEncoder().encode(candidate);
      const b = new TextEncoder().encode(owner?.password_hash || "0".repeat(64));
      if (!crypto.subtle.timingSafeEqual(a, b) || !owner) throw new HTTPException(401, { message: "Invalid email or password." });
    }
    const token = secret("tls_session_");
    await c.env.DB.prepare("INSERT INTO sessions(hash,account_id,expires_at) VALUES(?,?,?)")
      .bind(await hash(token), owner.id, now() + 30 * 86400).run();
    setCookie(c, "__Host-telos_session", token, { httpOnly: true, secure: true, sameSite: "Strict", path: "/", maxAge: 30 * 86400 });
    // CLI callers receive the same session credential and store it in an owner-only file.
    const verification_sent = action === "signup" ? await sendAction(c.env, { id: owner.id, email, salt: owner.salt }, "verify") : undefined;
    return c.json({ id: owner.id, email, token, verification_sent });
  });
}

app.route("/api", recovery);

app.use("/api/*", async (c, next) => {
  c.set("account", await account(c));
  await next();
});

app.get("/api/me", c => c.json(c.get("account")));
app.post("/api/logout", async c => {
  const token = c.req.header("authorization")?.replace(/^Bearer /, "") || getCookie(c, "__Host-telos_session");
  if (token) await c.env.DB.prepare("DELETE FROM sessions WHERE hash=?").bind(await hash(token)).run();
  deleteCookie(c, "__Host-telos_session", { secure: true, path: "/" });
  return c.json({ ok: true });
});

app.get("/api/keys", async c => c.json((await c.env.DB.prepare(
  "SELECT id,name,prefix,created_at,revoked_at FROM api_keys WHERE account_id=? ORDER BY created_at DESC",
).bind(c.get("account").id).all()).results));

async function keyValues(c: Parameters<typeof body>[0], name: string) {
  const key = secret("tls_key_");
  return { id: crypto.randomUUID(), key, values: [c.get("account").id, name, await hash(key), key.slice(0, 16), now()] };
}

app.post("/api/keys", async c => {
  requireVerified(c);
  const data = await body(c);
  const key = await keyValues(c, text(data.name, "key name", 80));
  await c.env.DB.prepare("INSERT INTO api_keys(id,account_id,name,hash,prefix,created_at) VALUES(?,?,?,?,?,?)")
    .bind(key.id, ...key.values).run();
  return c.json({ id: key.id, key: key.key }, 201);
});

app.post("/api/keys/:id/rotate", async c => {
  const old = await c.env.DB.prepare("SELECT name FROM api_keys WHERE id=? AND account_id=? AND revoked_at IS NULL")
    .bind(c.req.param("id"), c.get("account").id).first<{ name: string }>();
  if (!old) throw new HTTPException(404, { message: "Active API key not found." });
  const key = await keyValues(c, old.name);
  const operations = await c.env.DB.batch([
    c.env.DB.prepare("INSERT INTO api_keys(id,account_id,name,hash,prefix,created_at) SELECT ?,?,?,?,?,? WHERE EXISTS(SELECT 1 FROM api_keys WHERE id=? AND account_id=? AND revoked_at IS NULL)")
      .bind(key.id, ...key.values, c.req.param("id"), c.get("account").id),
    c.env.DB.prepare("UPDATE api_keys SET revoked_at=? WHERE id=? AND account_id=? AND revoked_at IS NULL")
      .bind(now(), c.req.param("id"), c.get("account").id),
  ]);
  if (!operations[0].meta.changes) throw new HTTPException(409, { message: "Key was already rotated or revoked." });
  return c.json({ id: key.id, key: key.key });
});

app.delete("/api/keys/:id", async c => {
  const result = await c.env.DB.prepare("UPDATE api_keys SET revoked_at=COALESCE(revoked_at,?) WHERE id=? AND account_id=?")
    .bind(now(), c.req.param("id"), c.get("account").id).run();
  if (!result.meta.changes) throw new HTTPException(404, { message: "API key not found." });
  return c.json({ ok: true });
});

app.get("/api/hosts", async c => c.json((await c.env.DB.prepare(
  "SELECT slug,model,shared,last_seen,created_at FROM hosts WHERE account_id=? ORDER BY slug",
).bind(c.get("account").id).all()).results));

app.post("/api/hosts", async c => {
  requireVerified(c);
  const data = await body(c);
  const slug = text(data.slug, "endpoint name", 40).toLowerCase();
  if (!/^[a-z0-9](?:[a-z0-9-]{0,38}[a-z0-9])?$/.test(slug) ||
      ["www", "api", "admin", "login", "signup", "account", "origin", "status"].includes(slug)) {
    throw new HTTPException(400, { message: "Choose an endpoint name using letters, digits, and internal hyphens." });
  }
  const model = text(data.model, "model ID");
  const result = await c.env.DB.prepare(
    "INSERT INTO hosts(slug,id,account_id,model,origin_key,created_at) VALUES(?,?,?,?,?,?) ON CONFLICT(slug) DO NOTHING",
  ).bind(slug, crypto.randomUUID(), c.get("account").id, model, secret("tls_origin_"), now()).run();
  if (!result.meta.changes) throw new HTTPException(409, { message: "That endpoint name is already reserved." });
  return c.json({ slug, model, public_url: `https://${c.env.MODEL_DOMAIN}/${slug}/v1` }, 201);
});

app.patch("/api/hosts/:slug", async c => {
  const host = await ownedHost(c, c.req.param("slug"));
  const data = await body(c);
  if (typeof data.shared !== "boolean") throw new HTTPException(400, { message: "shared must be true or false." });
  await c.env.DB.prepare("UPDATE hosts SET shared=? WHERE slug=?").bind(Number(data.shared), host.slug).run();
  return c.json({ slug: host.slug, shared: data.shared });
});

app.post("/api/hosts/:slug/connect", async c => {
  const host = await ownedHost(c, c.req.param("slug"));
  const data = await body(c);
  if (!Number.isInteger(data.port) || Number(data.port) < 1024 || Number(data.port) > 65535) {
    throw new HTTPException(400, { message: "Choose a local port between 1024 and 65535." });
  }
  if (data.model !== host.model) throw new HTTPException(409, { message: "Serve the model ID registered for this endpoint." });
  return c.json(await connect(c, host, Number(data.port)));
});

app.post("/api/hosts/:slug/heartbeat", async c => {
  const host = await ownedHost(c, c.req.param("slug"));
  const data = await body(c);
  if (!host.tunnel_id || data.connection_id !== host.connection_id) {
    throw new HTTPException(409, { message: "This endpoint connection has expired or been replaced." });
  }
  const health = await fetch("https://" + originHost(c.env, host) + "/health", {
    headers: { Authorization: "Bearer " + host.origin_key },
    signal: AbortSignal.timeout(5000), redirect: "manual",
  });
  await health.body?.cancel();
  if (!health.ok) throw new HTTPException(503, { message: "Waiting for the model tunnel to become reachable." });
  await c.env.DB.prepare("UPDATE hosts SET last_seen=?,connection_expires=? WHERE slug=? AND connection_id=?")
    .bind(now(), now()+90, host.slug, data.connection_id).run();
  return c.json({ ok: true });
});

app.post("/api/hosts/:slug/disconnect", async c => {
  const host = await ownedHost(c, c.req.param("slug"));
  const data = await body(c);
  if (typeof data.connection_id !== "string") throw new HTTPException(400, { message: "A connection ID is required." });
  await c.env.DB.prepare("UPDATE hosts SET last_seen=0,connection_id=NULL,connection_expires=0 WHERE slug=? AND connection_id=?")
    .bind(host.slug, data.connection_id).run();
  return c.json({ ok: true });
});

const inference: Handler<AppEnv> = async c => {
  const owner = await inferenceAccount(c);
  const hostname = new URL(c.req.url).hostname;
  if (hostname !== c.env.MODEL_DOMAIN && hostname !== "localhost") throw new HTTPException(404, { message: "Unknown model endpoint." });
  const slug = c.req.param("slug") || null;
  const pool = slug === null;
  const inferencePath = pool ? c.req.path : c.req.path.slice(slug.length + 1);
  if (inferencePath === "/v1/models" && c.req.method === "GET") {
    const hosts = pool
      ? (await c.env.DB.prepare("SELECT * FROM hosts WHERE (account_id=? OR shared=1) AND last_seen>?").bind(owner, now()-90).all<Host>()).results
      : [await c.env.DB.prepare("SELECT * FROM hosts WHERE slug=? AND (account_id=? OR shared=1) AND last_seen>?").bind(slug, owner, now()-90).first<Host>()].filter((h): h is Host => h !== null);
    return c.json({ object: "list", data: hosts.map(h => ({ id: pool ? h.slug + "/" + h.model : h.model, object: "model", owned_by: h.slug })) });
  }
  if (!["/v1/chat/completions", "/v1/completions"].includes(inferencePath) || c.req.method !== "POST") {
    throw new HTTPException(404, { message: "Inference endpoint not found." });
  }
  const data = await body(c);
  const requested = text(data.model, "model");
  const chosen = pool ? requested.split("/")[0] : slug;
  const host = await c.env.DB.prepare("SELECT * FROM hosts WHERE slug=? AND (account_id=? OR shared=1) AND last_seen>?")
    .bind(chosen, owner, now()-90).first<Host>();
  if (!host || requested !== (pool ? host.slug + "/" + host.model : host.model)) {
    throw new HTTPException(404, { message: "Model is unavailable or private." });
  }
  const origin = "https://" + originHost(c.env, host);
  const response = await fetch(origin + inferencePath, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer " + host.origin_key },
    body: JSON.stringify({ ...data, model: host.model }),
    signal: c.req.raw.signal,
    redirect: "manual",
  });
  if (response.status >= 300 && response.status < 400) {
    await response.body?.cancel();
    throw new HTTPException(502, { message: "Model origin redirects are not allowed." });
  }
  const headers = new Headers(response.headers);
  headers.delete("set-cookie");
  headers.set("Cache-Control", "no-store");
  return new Response(response.body, { status: response.status, headers });
};
app.all("/v1/*", inference);
app.all("/:slug/v1/*", inference);

app.get("*", c => {
  const hostname = new URL(c.req.url).hostname;
  if (hostname === c.env.MODEL_DOMAIN || hostname.endsWith("." + c.env.MODEL_DOMAIN)) {
    return c.json({ service: "Telos shared model hosting", base_url: `https://${c.env.MODEL_DOMAIN}/v1`, accounts: c.env.ACCOUNT_ORIGIN });
  }
  return c.env.ASSETS.fetch(c.req.raw);
});

export default app;
