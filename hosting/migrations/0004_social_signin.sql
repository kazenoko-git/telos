ALTER TABLE sessions ADD COLUMN authenticated_at INTEGER NOT NULL DEFAULT 0;
CREATE TABLE oauth_identities (
  provider TEXT NOT NULL CHECK(provider IN ('google','github')),
  subject TEXT NOT NULL,
  account_id TEXT NOT NULL REFERENCES accounts(id),
  created_at INTEGER NOT NULL,
  PRIMARY KEY(provider,subject),
  UNIQUE(account_id,provider)
);
CREATE TABLE oauth_flows (
  state_hash TEXT PRIMARY KEY,
  browser_hash TEXT NOT NULL,
  provider TEXT NOT NULL CHECK(provider IN ('google','github')),
  verifier TEXT NOT NULL,
  nonce TEXT NOT NULL,
  account_id TEXT REFERENCES accounts(id),
  session_hash TEXT,
  credential_salt TEXT,
  expires_at INTEGER NOT NULL
);
CREATE INDEX oauth_owner ON oauth_identities(account_id);
