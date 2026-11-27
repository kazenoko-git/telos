# Account-based model hosting

Accounts and key management: https://telos.wingit.tech.
Reserved endpoint: `https://tls.wingit.tech/<username>/v1`.
The username is a reserved endpoint name (CLI argument `slug`), owned by the signed-in account. An account can reserve multiple names for different models. It is not derived from the email address.

```sh
telos account signup --email you@example.com
telos keys create laptop
telos tunnel reserve alice --model my-model
telos serve --checkpoint /path/to/checkpoint --model-name my-model --tunnel alice
```

Install Cloudflare's `cloudflared` connector first. Serving binds to loopback; the connector establishes the outbound tunnel. Stop the server to disconnect it. Restarting the server keeps the reserved path and consumer API keys.

Configure an OpenAI-compatible client with base URL `https://tls.wingit.tech/alice/v1`, model `my-model`, and a persistent API key created in the portal or CLI. Management sessions expire after 30 days. API keys remain valid until rotated or revoked; full keys are shown only at creation. Rotation invalidates the old key. Revocation takes effect on new requests, including requests for shared models; it does not terminate streams already in progress.

Endpoints are private by default. `telos tunnel share alice` explicitly allows other Telos accounts to send prompts to that model; `telos tunnel unshare alice` restores private access. The shared pool remains available at `https://tls.wingit.tech/v1`, where model IDs are `alice/my-model`. Per-owner paths use the original model ID. Consumer keys and provider origin credentials are separate.

## Hosting operation

The Worker routes only `telos.wingit.tech/*` and `tls.wingit.tech/*`. Path-based endpoints use the existing certificate and do not consume individual Worker Custom Domains. Cloudflare free-plan request, CPU, database and tunnel limits still apply; this is not a promise of unlimited free traffic.

The hosting administrator configures the account and zone IDs, D1 database and `CLOUDFLARE_API_TOKEN` secret. The scoped token needs wingit.tech DNS Write and account cloudflared Connector Write. Workers domain-management permissions are unnecessary. Never commit the token or expose it to model consumers. Run the D1 migrations before deployment. Migration 0002 records an unused column from the individual-domain pilot; retain it for databases where it has already been applied.

This release implements password-based signup/login, reservations, key lifecycle and explicit sharing. Email verification, password recovery and per-provider billing are not implemented.

## Validation (2026-11-27)

Eight Workers fixture tests cover account/key ownership, rotation races, reservations, sharing, streaming, connection leases, provisioning recovery, path routing and redirect rejection. Thirteen Python tests cover the client, connector lifecycle and serving compatibility.

A live CPU-only tiny checkpoint (two layers, width 64, context 32) was served through Cloudflare over verified HTTPS. Both streaming and non-streaming four-token generations matched direct local generation exactly. Owner access, private isolation, explicit sharing, consumer output parity, rotation and revocation passed. The reserved path returned HTTP 200 and exact output. This validates the hosting transport, not real-model quality or production load capacity.
