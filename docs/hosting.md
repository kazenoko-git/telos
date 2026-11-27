# Account-based model hosting

Accounts and key management: https://telos.wingit.tech/account.
The research homepage stays at https://telos.wingit.tech.
Reserved endpoint: `https://tls.wingit.tech/<username>/v1`.
The username is a reserved endpoint name (CLI argument `slug`), owned by the signed-in account. An account can reserve multiple names for different models. It is not derived from the email address.

```sh
telos account signup --email you@example.com

# Open the signup email and select Verify email before continuing.
telos keys create laptop
telos tunnel reserve alice --model my-model
telos serve --checkpoint /path/to/checkpoint --model-name my-model --tunnel alice
```

Install Cloudflare's `cloudflared` connector first. Serving binds to loopback; the connector establishes the outbound tunnel. Stop the server to disconnect it. Restarting the server keeps the reserved path and consumer API keys.

Configure an OpenAI-compatible client with base URL `https://tls.wingit.tech/alice/v1`, model `my-model`, and a persistent API key created in the portal or CLI. Management sessions expire after 30 days. API keys remain valid until rotated or revoked; full keys are shown only at creation. Rotation invalidates the old key. Revocation takes effect on new requests, including requests for shared models; it does not terminate streams already in progress.

Endpoints are private by default. `telos tunnel share alice` explicitly allows other Telos accounts to send prompts to that model; `telos tunnel unshare alice` restores private access. The shared pool remains available at `https://tls.wingit.tech/v1`, where model IDs are `alice/my-model`. Per-owner paths use the original model ID. Consumer keys and provider origin credentials are separate.

## Hosting operation

The Worker routes only the account page, `/api/*`, `/hosting-assets/*` and `tls.wingit.tech/*`. The research homepage and its assets remain served by the existing Cloudflare Pages site. Path-based endpoints use the existing certificate and do not consume individual Worker Custom Domains. Cloudflare free-plan request, CPU, database and tunnel limits still apply; this is not a promise of unlimited free traffic.

The hosting administrator configures the account and zone IDs, D1 database and `CLOUDFLARE_API_TOKEN` secret. The scoped token needs wingit.tech DNS Write and account cloudflared Connector Write. Workers domain-management permissions are unnecessary. Never commit the token or expose it to model consumers. Run the D1 migrations before deployment. Migration 0002 records an unused column from the individual-domain pilot; retain it for databases where it has already been applied.

This release implements password-based signup/login, email verification, password recovery, reservations, key lifecycle and explicit sharing. Per-provider billing is not implemented.

New key creation and endpoint reservations require email verification. Existing accounts start unverified; their existing keys and endpoints continue working. Verification links expire after 24 hours, reset links after 30 minutes. Both are single-use, purpose-bound random tokens stored only as hashes. Password reset invalidates all management sessions and older email links while preserving API keys. API keys can be separately rotated or revoked from the portal. Because CLI-hosted tunnels use management sessions for heartbeats, a password reset also requires signing the CLI in again and restarting its server; the reserved URL and consumer keys are unchanged.

Use **Forgot password?** after entering your email in the sign-in form. The response does not disclose whether an account exists. Signed-in users can resend verification emails. Email requests are limited by IP and destination; links use URL fragments that the portal removes before submitting tokens in POST bodies. Opening a link does not consume it until the user confirms. Failed mail sends discard their tokens.

The sender is `accounts@telos.wingit.tech`, using the native EMAIL binding and onboarded subdomain. Cloudflare Workers Free only delivers to account-verified destination addresses; arbitrary public signup recipients require Workers Paid or a different mail provider. No billing upgrade was performed. A successfully accepted send is not proof of inbox delivery.

## Validation (2026-11-27)

Fourteen Workers fixture tests cover account/key ownership, rotation races, reservations, sharing, streaming, connection leases, provisioning recovery, path routing, redirect rejection, verification, reset races, token expiry and reuse, delivery failure, session invalidation and legacy API-key continuity. Thirteen Python tests cover the client, connector lifecycle and serving compatibility.

A live CPU-only tiny checkpoint (two layers, width 64, context 32) was served through Cloudflare over verified HTTPS. Both streaming and non-streaming four-token generations matched direct local generation exactly. Owner access, private isolation, explicit sharing, consumer output parity, rotation and revocation passed. The reserved path returned HTTP 200 and exact output. This validates the hosting transport, not real-model quality or production load capacity.

To repeat the full fresh-account CLI smoke test explicitly:

```sh
.venv/bin/python scripts/check_hosting_flow.py --email disposable-inbox@example.com --output reports/hosting_flow.json
```

This creates a disposable live account, key and reserved name, starts and restarts a tiny CPU server, and checks the same public URL and key after restart. It checks streaming parity, rotates the key, verifies the old key is rejected, checks generation with the replacement, and verifies revocation. It then stops its server, revokes its test keys and logs out. The report contains boolean results only; credentials and temporary checkpoints are not retained. The disposable account and offline reservation remain in the hosting database.

Live email recovery was tested with the user-provided disposable inbox on 2026-11-27. The user completed the verification and reset forms and confirmed the new-password login. Independent server checks confirmed verification, rejection of the old session and password (401), and continued authorization by the same API key (200). The test key was then revoked and its private local credentials removed. No real model was loaded.

The original research homepage was restored at the root. HTTP checks confirmed the research title at `/`, the portal title at `/account` without a redirect, and the isolated portal script under `/hosting-assets/`.
