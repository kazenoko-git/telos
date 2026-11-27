ALTER TABLE accounts ADD COLUMN email_verified_at INTEGER;
CREATE TABLE account_actions (
  hash TEXT PRIMARY KEY,
  account_id TEXT NOT NULL REFERENCES accounts(id),
  purpose TEXT NOT NULL CHECK(purpose IN ('verify','reset')),
  credential_salt TEXT NOT NULL,
  expires_at INTEGER NOT NULL
);
CREATE INDEX actions_owner ON account_actions(account_id,purpose);
