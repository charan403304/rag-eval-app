# Database Migration Policy

## Writing migrations

Migrations live in `migrations/` and are numbered sequentially, e.g.
`0042_add_orders_index.sql`. Every migration must be reversible: pair each
`up` migration with a corresponding `down` migration that undoes it
cleanly. Migrations that are not reversible (e.g. destructive data changes)
require sign-off from a second engineer before merging.

## Backward compatibility

Because we do rolling deploys, the old code and the new code run against
the same database simultaneously for a short window. This means a
migration must never break the currently-running (old) code. In practice
this means: add columns as nullable first, backfill in a separate step,
then add a `NOT NULL` constraint in a later migration once backfill is
confirmed complete. Never rename or drop a column in the same migration
that stops using it — do it in two separate deploys, at least one release
apart.

## Running migrations

Migrations run automatically as part of the deploy pipeline, before the
new application code starts serving traffic. If a migration fails, the
deploy is aborted and the previous version keeps serving. Migrations
timeout after 5 minutes; long-running migrations (e.g. backfills on large
tables) should be written as idempotent scripts run manually in batches,
not as part of the automatic migration step.

## Testing migrations

Every migration is tested against a snapshot of production schema (not
production data) in CI before merge. This catches most syntax and
constraint issues but does not catch performance problems on large tables,
which should be checked manually via `EXPLAIN ANALYZE` on a staging replica
before running in production.
