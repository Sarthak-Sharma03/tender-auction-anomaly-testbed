# Limitations (on purpose)

- Quickstart uses `failed_events > 0` (any failed login in the session) as a proxy risk signal. It is useful to
  sanity-check ranking under budgets, not to claim state-of-the-art accuracy.
- The public dataset is a surrogate for the tender/auction domain (real production data is under NDA).
  See `docs/DOMAIN_CONTEXT.md` for the mapping.
- Sessionization is user-day. It is simple and stable for review; adapt it if your operational definition of a
  session differs.
- Drift check is PSI + KS statistic only (fast + dependency-light). In real work you would add richer monitoring
  (score drift, feature KS tests, alert-volume drift).
- Confidence bounds and risk curves are proxy-based; replace the proxy label with real labels for claims.
- `failed_flagged` in the report is a rough severity indicator (total failed events inside flagged sessions).
