import { HTTPException } from "hono/http-exception";
import type { Context } from "hono";
import { now, type AppEnv } from "./auth";

export type Host = {
  slug: string; id: string; account_id: string; model: string; shared: number;
  origin_key: string; tunnel_id: string | null; port: number | null; last_seen: number;
  connection_id: string | null; connection_expires: number;
};

export async function ownedHost(c: Context<AppEnv>, slug: string): Promise<Host> {
  const host = await c.env.DB.prepare("SELECT * FROM hosts WHERE slug=? AND account_id=?")
    .bind(slug, c.get("account").id).first<Host>();
  if (!host) throw new HTTPException(404, { message: "Reserved endpoint not found." });
  return host;
}

type CFResult<T> = { success: boolean; result: T };

async function cf<T>(env: Env, path: string, method = "GET", data?: unknown): Promise<T> {
  if (!env.CLOUDFLARE_API_TOKEN || !env.CF_ZONE_ID) {
    throw new HTTPException(503, { message: "Tunnel provisioning is awaiting the hosting administrator's Cloudflare configuration." });
  }
  const response = await fetch("https://api.cloudflare.com/client/v4" + path, {
    method, headers: { Authorization: "Bearer " + env.CLOUDFLARE_API_TOKEN, "Content-Type": "application/json" },
    body: data === undefined ? undefined : JSON.stringify(data),
    signal: AbortSignal.timeout(15000),
  });
  if (!response.ok) throw new HTTPException(502, { message: `Cloudflare provisioning returned ${response.status}. Retry after checking configuration.` });
  const result = await response.json<CFResult<T>>();
  if (!result.success) throw new HTTPException(502, { message: "Cloudflare could not provision this tunnel." });
  return result.result;
}

export function originHost(env: Env, host: Host): string {
  return `tlso-${host.id}.${env.ORIGIN_ZONE}`;
}

export async function connect(c: Context<AppEnv>, host: Host, port: number) {
  const connectionId = crypto.randomUUID();
  const locked = await c.env.DB.prepare(
    "UPDATE hosts SET lease_until=?,connection_id=?,connection_expires=? WHERE slug=? AND lease_until<? AND connection_expires<?",
  ).bind(now() + 180, connectionId, now()+180, host.slug, now(), now()).run();
  if (!locked.meta.changes) throw new HTTPException(409, { message: "This endpoint already has an active server or is connecting. Stop it or wait for its connection lease to expire." });
  try {
    const accountPath = `/accounts/${c.env.CF_ACCOUNT_ID}/cfd_tunnel`;
    let tunnelId = host.tunnel_id;
    if (!tunnelId) {
      const name = "telos-" + host.id;
      const existing = await cf<{ id: string }[]>(c.env, accountPath + "?is_deleted=false&name=" + name);
      const tunnel = existing[0] || await cf<{ id: string }>(c.env, accountPath, "POST", { name, config_src: "cloudflare" });
      tunnelId = tunnel.id;
      await c.env.DB.prepare("UPDATE hosts SET tunnel_id=? WHERE slug=?").bind(tunnelId, host.slug).run();
    }
    const origin = originHost(c.env, host);
    await cf(c.env, accountPath + "/" + tunnelId + "/configurations", "PUT", {
      config: { ingress: [{ hostname: origin, service: `http://127.0.0.1:${port}` }, { service: "http_status:404" }] },
    });
    const dnsPath = `/zones/${c.env.CF_ZONE_ID}/dns_records`;
    const records = await cf<{ id: string; type: string; content: string }[]>(c.env, dnsPath + "?name=" + origin);
    if (records.some(r => r.type !== "CNAME" || r.content !== tunnelId + ".cfargotunnel.com")) {
      throw new HTTPException(409, { message: "The origin DNS name already belongs to another service." });
    }
    if (!records.length) await cf(c.env, dnsPath, "POST", {
      type: "CNAME", name: origin, content: tunnelId + ".cfargotunnel.com", proxied: true, ttl: 1,
    });
    const token = await cf<string>(c.env, accountPath + "/" + tunnelId + "/token");
    await c.env.DB.prepare("UPDATE hosts SET port=?,last_seen=0 WHERE slug=?").bind(port, host.slug).run();
    return { token, connection_id: connectionId, origin_key: host.origin_key, public_url: `https://${host.slug}.${c.env.MODEL_DOMAIN}/v1` };
  } catch (error) {
    await c.env.DB.prepare("UPDATE hosts SET connection_id=NULL,connection_expires=0 WHERE slug=? AND connection_id=?")
      .bind(host.slug, connectionId).run();
    throw error;
  } finally {
    await c.env.DB.prepare("UPDATE hosts SET lease_until=0 WHERE slug=?").bind(host.slug).run();
  }
}
