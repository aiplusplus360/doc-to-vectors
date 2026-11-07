# Production Handbook

This handbook is the source of truth for running the payments API. It is written
for on-call engineers who need an answer in under a minute.

## Deployments

Deploys are triggered from `main` only. A deploy that starts after 16:00 UTC
needs an on-call engineer watching the dashboards for thirty minutes.

### Standard deploy

The standard deploy runs migrations, then rolls pods in two batches.

```bash
./bin/deploy --env production --batch-size 2
./bin/deploy status
```

If a batch does not report healthy within five minutes, the deploy halts and
the previous revision keeps serving traffic.

### Rollback

Rollback is always safe because migrations are additive for one release.

```bash
./bin/deploy rollback --env production --to previous
```

Do not roll back across a migration that dropped a column. Check the migration
log first.

## Incident response

Severity levels decide who gets paged and how fast.

| Severity | Definition | Page | Update cadence |
| --- | --- | --- | --- |
| SEV1 | Payments failing for all users | Immediately | Every 15 minutes |
| SEV2 | Payments failing for one region | Immediately | Every 30 minutes |
| SEV3 | Degraded but working | Business hours | Daily |
| SEV4 | Cosmetic | Ticket | Weekly |

### First five minutes

1. Acknowledge the page so the next responder knows it is owned.
2. Open the runbook for the affected service.
3. Post in the incident channel with severity and a one-line summary.
4. Only then start investigating.

Guessing at a cause before acknowledging the page wastes the most valuable
minutes of an incident.

## Secrets

Signing keys live in the key vault and are rotated every 90 days.

### Rotating the signing key

Rotation is a two-phase operation: publish the new key, wait for every verifier
to pick it up, then retire the old key.

```bash
vault rotate signing-key --staged
vault status signing-key
```

Retiring the old key early invalidates in-flight tokens and shows up as a spike
in 401 responses.

## Runbooks

Each service has a runbook with the same four sections: symptoms, dashboards,
first checks, escalation. If a runbook is missing a section, that is a bug.
