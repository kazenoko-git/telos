-- Retained migration history from the individual-domain pilot.
-- Path-based hosting does not use this column.
ALTER TABLE hosts ADD COLUMN domain_id TEXT;
