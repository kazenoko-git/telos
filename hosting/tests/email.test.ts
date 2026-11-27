import { env } from "cloudflare:workers";
import { afterEach, expect, it, vi } from "vitest";
import { sendAccountMail } from "../src/email";

const message = { to: "disposable@example.invalid", subject: "Verify your email", text: "fixture link", html: "<p>fixture link</p>" };
const configured = () => ({ ...env, EMAIL_PROVIDER: "resend", RESEND_API_KEY: "fixture-sending-key" });
afterEach(() => vi.restoreAllMocks());

it("sends through Resend with bounded time, no redirects and an idempotency key", async () => {
  const send = vi.spyOn(globalThis, "fetch").mockResolvedValue(Response.json({ id: "fixture-message" }));
  await sendAccountMail(configured(), message, "fixture-operation");
  expect(send).toHaveBeenCalledOnce();
  const [url, options] = send.mock.calls[0];
  expect(url).toBe("https://api.resend.com/emails");
  expect(options?.headers).toMatchObject({ Authorization: "Bearer fixture-sending-key", "Idempotency-Key": "fixture-operation" });
  expect(options?.redirect).toBe("manual");
  expect(options?.signal).toBeDefined();
  expect(JSON.parse(String(options?.body))).toMatchObject({ to: [message.to], subject: message.subject, text: message.text });
});

it.each([302,401,429,500])("rejects Resend HTTP %i without falling back to another provider", async status => {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("provider failure", { status }));
  const other = vi.spyOn(env.EMAIL, "send");
  await expect(sendAccountMail(configured(), message, "fixture-operation")).rejects.toThrow("Email delivery rejected");
  expect(other).not.toHaveBeenCalled();
});

it("fails closed for missing configuration or a malformed acceptance", async () => {
  const send = vi.spyOn(globalThis, "fetch").mockResolvedValue(Response.json({}));
  await expect(sendAccountMail({ ...configured(), RESEND_API_KEY: "" }, message, "fixture-operation")).rejects.toThrow("not configured");
  expect(send).not.toHaveBeenCalled();
  await expect(sendAccountMail(configured(), message, "fixture-operation")).rejects.toThrow("did not accept");
});
