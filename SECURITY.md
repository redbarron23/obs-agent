# Security

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/redbarron23/obs-agent/security/advisories/new),
not in a public issue. I aim to reply within a week.

## Scope and design notes

- **Data.** The repository and the hosted demo use synthetic data only. No real cloud
  credentials, billing exports or customer data are included or needed.
- **Hosted demo** (`obs-agent.streamlit.app`, `OBS_AGENT_DEMO=1`): no LLM is configured and
  the chat input is refused server-side, so visitors cannot trigger model calls. Review
  decisions go to a per-session SQLite file under a random name and are never shared between
  visitors.
- **REST API.** All routes except health require a configured `OBS_AGENT_API_KEY`. The API
  refuses protected requests when no key is configured: `/ask` spends your LLM provider credits. See
  [docs/api.md](docs/api.md).
- **Private imports.** Local saved reviews retain uploaded CSVs, vocabularies, and approvals on
  disk. Keep `OBS_AGENT_REVIEW_DIR` private and outside version control. Users of one installation
  share its saved reviews; it is intended for one trusted team, not multiple tenants.
- **Reviewer identity.** Reviewer names/IDs in the UI and API are self-reported. Authenticate
  reviewers at the deployment boundary before treating them as verified identities.
- **Exports.** The generated `az` / `gcloud` script quotes every value with `shlex.quote`, but
  review it before running it: it changes tags on real resources.
- **Secrets.** API keys come from the environment (`.env` is git-ignored). Secret scanning
  and push protection are enabled on this repository.
