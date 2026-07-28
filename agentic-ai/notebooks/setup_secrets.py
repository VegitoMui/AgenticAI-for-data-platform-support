# Databricks notebook source
# MAGIC %md
# MAGIC # Setup: Secret Scope
# MAGIC
# MAGIC Run this once per workspace, inside Databricks. Replaces any local CLI setup.
# MAGIC
# MAGIC Values are entered through widgets, written to a secret scope, then cleared.
# MAGIC Nothing is committed and nothing is printed back.

# COMMAND ----------

# MAGIC %pip install databricks-sdk --quiet
# MAGIC %restart_python

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceAlreadyExists

w = WorkspaceClient()

SCOPE = "agentic-ai"

SECRET_KEYS = [
    ("openrouter-api-key", "OpenRouter API key (primary LLM)"),
    ("groq-api-key-1",     "Groq API key 1 (fallback)"),
    ("groq-api-key-2",     "Groq API key 2 (fallback)"),
    ("smtp-user",          "SMTP username"),
    ("smtp-password",      "SMTP app password"),
    ("smtp-sender",        "From address on notifications"),
    ("approver-email",     "Where approval notifications are sent"),
    ("github-owner",       "GitHub org or user"),
    ("github-repo",        "GitHub repository name"),
    ("github-token",       "GitHub token: Issues read/write, Contents read"),
    ("app-base-url",       "Approvals App URL (leave blank until Phase 2)"),
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the scope

# COMMAND ----------

try:
    w.secrets.create_scope(scope=SCOPE)
    print(f"created scope: {SCOPE}")
except ResourceAlreadyExists:
    print(f"scope already exists: {SCOPE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enter values
# MAGIC
# MAGIC Fill the widgets at the top of the notebook, then run the next cell.
# MAGIC Blank widgets are skipped, so you can run this repeatedly to set or
# MAGIC rotate a subset of keys.

# COMMAND ----------

dbutils.widgets.removeAll()
for key, label in SECRET_KEYS:
    dbutils.widgets.text(key, "", label)

print("Widgets created. Fill them in above, then run the next cell.")

# COMMAND ----------

written, skipped = [], []

for key, _ in SECRET_KEYS:
    value = dbutils.widgets.get(key).strip()
    if not value:
        skipped.append(key)
        continue
    w.secrets.put_secret(scope=SCOPE, key=key, string_value=value)
    written.append(key)

print(f"written ({len(written)}): {', '.join(written) or 'none'}")
print(f"skipped ({len(skipped)}): {', '.join(skipped) or 'none'}")

# Clear the widgets so values do not persist in the notebook state.
dbutils.widgets.removeAll()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify
# MAGIC
# MAGIC Lists key names only. Values are never readable back through the API.

# COMMAND ----------

for s in w.secrets.list_secrets(scope=SCOPE):
    print(f"  {s.key}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grant read access to the job identity
# MAGIC
# MAGIC Replace the principal below with the service principal that runs the
# MAGIC watcher and reconciler jobs.

# COMMAND ----------

from databricks.sdk.service.workspace import AclPermission

PRINCIPAL = "REPLACE_WITH_SERVICE_PRINCIPAL_APPLICATION_ID"

if not PRINCIPAL.startswith("REPLACE"):
    w.secrets.put_acl(scope=SCOPE, principal=PRINCIPAL, permission=AclPermission.READ)
    print(f"granted READ on {SCOPE} to {PRINCIPAL}")
else:
    print("set PRINCIPAL above before running this cell")
