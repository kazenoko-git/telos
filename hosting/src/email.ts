export type AccountMail = { to: string; subject: string; text: string; html: string };

export async function sendAccountMail(env: Env, message: AccountMail, idempotencyKey: string): Promise<void> {
  if (env.EMAIL_PROVIDER === "cloudflare") {
    await env.EMAIL.send({ ...message, from: { email: env.EMAIL_FROM, name: "Telos" } });
    return;
  }
  if (env.EMAIL_PROVIDER !== "resend" || !env.RESEND_API_KEY) throw new Error("Email provider is not configured");
  const response = await fetch("https://api.resend.com/emails", {
    method: "POST", redirect: "manual", signal: AbortSignal.timeout(15000),
    headers: { Authorization: "Bearer " + env.RESEND_API_KEY, "Content-Type": "application/json", "Idempotency-Key": idempotencyKey },
    body: JSON.stringify({ ...message, from: `Telos <${env.EMAIL_FROM}>`, to: [message.to] }),
  });
  // Never forward credentials on redirects or log the response (which can include recipients).
  if (!response.ok) { await response.body?.cancel(); throw new Error("Email delivery rejected"); }
  const result = await response.json<{ id?: string }>();
  if (!result.id) throw new Error("Email provider did not accept the message");
}
