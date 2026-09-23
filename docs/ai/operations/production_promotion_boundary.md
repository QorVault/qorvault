# Production Promotion Boundary

This document defines the boundary between repo/dev work and
production-impacting work for QorVault.

It is a policy document only. It does not authorize production access,
deployment, promotion, rollback, Framework access, database writes,
migrations, ingestion, indexing, embedding, OSPI downloads, OSPI corpus
mutation, or live BoardDocs / Kent School District access.

## Current Main-Branch Boundary

As of Issue #39, `main` at
`0ba6cc053e66ead78c1deeddac5057860931a361` has one tracked GitHub
Actions workflow: `.github/workflows/stage1-no-secrets-ci.yml`.

That workflow runs Stage 1 No-Secrets CI on pull requests. It checks a
small, synthetic-safe Python surface and does not deploy, promote, write
databases, run migrations, run ingestion, run indexing, run embedding,
access Framework, download OSPI data, mutate the OSPI corpus, or contact
live BoardDocs / Kent School District services.

No tracked GitHub Actions workflow currently documents or implements an
automatic production deployment from `main`.

Therefore:

- Merging to `main` is not production unless a documented auto-deploy
  exists.
- CI green does not mean production deployed.
- Docs, planning, profile, and test work do not hit production by
  themselves.

If a future workflow, host service, branch protection rule, external
automation, or deployment system makes `main` auto-deploy, this document
must be updated before relying on the current boundary.

## Definitions

**Production** means any QorVault environment, service, database,
scheduled job, automation, credential set, or hosted surface whose change
can affect live users, live civic-record retrieval, live BoardDocs / Kent
School District polling, persistent production data, production
infrastructure, production secrets, or a public/runtime-facing answer
path.

Production also includes any action that promotes dev artifacts into a
live service, enables scheduled automation against live sources, changes
production environment variables or secrets, applies production database
schema or data changes, or performs rollback on a production service.

**Dev-only** means work performed against developer-controlled code,
synthetic data, local fixtures, non-production services, or explicitly
designated dev databases. Dev-only work may include docs, plans, local
tests, synthetic tests, profile reports, code review, and dev-only
verification. Dev-only work does not affect production unless a separate
owner-approved promotion step connects it to production.

**Local-only** means work performed inside a developer workstation or
local checkout without network access to live systems, without production
credentials, without production databases, and without changing any
remote service. Local-only work may create or inspect files in the local
repo, run safe local checks, or prepare a branch. It does not deploy or
promote anything by itself.

## What Requires Donald's Explicit Approval

The following actions require Donald's explicit approval before they
begin. A merged PR, green CI result, local success, or prior docs approval
is not enough.

- Production deployment.
- Production database writes.
- Production database migrations.
- Production secrets or environment changes.
- Framework access.
- Live scheduled automation.
- Live BoardDocs / KSD polling.
- OSPI ingestion beyond dev.
- Indexing or embedding runs against live, production, or promotion-bound
  data.
- Promotion or rollback of any service, job, model, index, corpus, or
  database state.

Approval must be specific to the action, target environment, data source,
expected side effects, operator, and rollback plan. Approval for one
action does not imply approval for adjacent actions.

## Evidence Required Before Promotion

Before any production promotion is considered, the operator must present
plain-English evidence for Donald's review. The evidence should be
specific enough that the project owner can see what is being promoted,
why it is ready, and how it can be reversed.

Required evidence:

- The relevant PRs are merged or otherwise explicitly approved for the
  promotion candidate.
- CI is green for the relevant branch or commit.
- Local or dev verification has been run against the exact behavior being
  promoted.
- Any migration, database-write, ingestion, indexing, embedding, or
  scheduled-automation change has a separate review of its data impact.
- Any production secret or environment change is listed by name and
  purpose, without exposing secret values.
- The target environment and affected services are identified.
- The expected production side effects are listed.
- A rollback plan exists and names the exact service, job, database,
  corpus, index, model, or configuration state to restore.
- A safety attestation confirms what was not touched during preparation.

Promotion evidence is a gate, not a deployment command. Presenting the
evidence does not authorize production action until Donald explicitly
approves the promotion.

## Rollback Expectations

Every production promotion must have a rollback plan before it starts.
The plan must identify:

- The previous known-good commit, image, configuration, model, index,
  corpus snapshot, database schema state, or service state.
- The operator responsible for rollback.
- The command or operational path that would perform rollback, described
  in plain English if commands are not yet approved.
- The data that may not be reversible, including database writes,
  migrations, ingestion runs, indexing runs, embedding runs, or source
  polling.
- The verification that proves rollback succeeded.

If rollback depends on production access, database access, Framework
access, secrets, migrations, ingestion, indexing, embedding, or live
source access, that access must be separately owner-approved before the
rollback is attempted.

## Safety Attestation Required Before Promotion

Before promotion, the operator must attest whether each statement is true
for the preparation work:

- No production access occurred.
- No Framework access occurred.
- No production database writes occurred.
- No production migrations were run.
- No production secrets or environment variables were changed.
- No ingestion was run beyond the approved dev scope.
- No indexing or embedding was run beyond the approved dev scope.
- No OSPI scripts were run unless separately approved for the stated
  scope.
- `download_ospi.py` was not run unless separately approved for the
  stated scope.
- No OSPI corpus files were mutated.
- No live BoardDocs / Kent School District polling or downloads occurred.
- No deployment, promotion, rollback, or live scheduled automation was
  performed.

If any statement is false, promotion must stop until Donald reviews the
exception and explicitly decides the next action.

## Non-Authorization

This document only defines gates and boundaries. It is not approval to
cross them.

Future work must continue to treat docs/planning/profile/test activity,
CI success, branch creation, commits, PRs, and merges as repo/dev events
unless a documented production automation path says otherwise and Donald
has approved the specific production action.
