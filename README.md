# Agentic AI Auto-Remediation

Autonomous detection, diagnosis, and human-approved remediation of Databricks
pipeline failures. Runs entirely in the cloud: two scheduled jobs and one
Databricks App on serverless, deployed by GitHub Actions. No interactive
notebooks in the runtime path, and no local machine in the deploy path.

## Where things run

| Concern | Runs on |
|---|---|
| Watcher, reconciler, migrate jobs | Databricks serverless |
| Approvals UI | Databricks App |
| Build, validate, deploy | GitHub Actions |
| Editing code | Databricks Git folder, github.dev, or Codespaces |
| Secrets setup and connectivity checks | `notebooks/` inside the workspace |

## Layout

```
databricks.yml                bundle definition, dev and prod targets
.github/workflows/deploy.yml  CI/CD: lint, test, validate, deploy
resources/jobs.yml            watcher, reconciler, migrate jobs, approvals app
notebooks/                    in-workspace setup and verification
src/agentic_ai/
  config.py                   Settings, resolved from args + secret scope
  secrets.py                  dbutils secrets with env fallback
  entrypoints.py              wheel entry points referenced by jobs.yml
  llm/                        multi-provider client with failover
  telemetry/                  system-table readers, watermark, watcher
  agents/                     the four agents
  subagents/                  sub-agent classifiers and tool routing
  remediation/                action classification, execution, approval, reconciler
  notify/                     email notification (link-only, no reply parsing)
  vcs/                        GitHub issues and commit history
  app/                        FastAPI approvals UI
```

## Setup

1. Push this repo to GitHub. Create `main` and `develop`.
2. Create a Databricks service principal with an OAuth secret. Grant it
   workspace access and the system table grants below.
3. Add repo secrets in GitHub: `DATABRICKS_HOST_DEV`, `DATABRICKS_CLIENT_ID_DEV`,
   `DATABRICKS_CLIENT_SECRET_DEV`, and the three `_PROD` equivalents.
4. Create GitHub environments `dev` and `prod`. Add a required reviewer on `prod`.
5. Set the workspace host in `databricks.yml` for both targets.
6. Push to `develop`. Actions validates and deploys.
7. In the workspace, run `notebooks/setup_secrets` then
   `notebooks/connectivity_check`.

## Configuration

No credential is ever read from a literal in source. `Settings` resolves every
secret through `dbutils.secrets` on a cluster, falling back to `AGENTIC_<KEY>`
environment variables otherwise.

Always build table names with `settings.table("name")` rather than f-strings.
It normalises to `catalog.schema.leaf` and prevents the double-prefix bug.

## Required grants

Run as a metastore admin, replacing the principal with the job service principal:

```sql
GRANT USE CATALOG ON CATALOG system TO `<principal>`;
GRANT USE SCHEMA, SELECT ON SCHEMA system.lakeflow TO `<principal>`;
GRANT USE SCHEMA, SELECT ON SCHEMA system.query    TO `<principal>`;
GRANT USE SCHEMA, SELECT ON SCHEMA system.access   TO `<principal>`;
```

Plus `USE CATALOG` / `USE SCHEMA` / `MODIFY` on the target schema, and `READ`
on the `agentic-ai` secret scope (granted by the last cell of `setup_secrets`).

GitHub token needs Issues read/write and Contents read.

## Phase status

| Phase | Scope | State |
|---|---|---|
| 0 | Repo, bundle, secrets, config, LLM client, GitHub client, CI/CD | done |
| 1 | Watcher against `system.lakeflow`, watermark, incident creation | next |
| 2 | Approval on every request, App UI, reconciler, GitHub issues | pending |
| 3 | Sub-agent routing, execution trace | pending |
| 4 | Duplicate-table, VACUUM, OPTIMIZE/ZORDER scenarios | pending |
| 5 | Analytics semantic model and dependency graph | pending |
