# Domain context and NDA note

This repo is a public testbed for a tender/auction anomaly detection pipeline that was delivered under NDA.
To keep the artifact reproducible and shareable, the public data is a login-like surrogate and the repo ships a
synthetic generator. The pipeline, evaluation, and engineering are unchanged.

## Field mapping (tender/auction -> testbed)

| Tender/auction concept | Testbed column | Notes |
|---|---|---|
| Bid or event timestamp | Login Timestamp | Event time driving sessionization |
| Bidder or supplier ID | User ID | Entity for per-actor sessions |
| Source IP / network origin | IP Address | Used for geo/device diversity features |
| Country / region | Country | Example geo field (can be richer in production) |
| Client or device fingerprint | Browser Name and Version, Device Type | Proxy for device/user agent changes |
| Rule violation / failed submission | Login Successful | Proxy risk signal; `failed_events` aggregates per session |

Sessionization in the testbed is bidder-day (user-day). In production, use your operational session definition
(e.g., tender window, sliding time windows, or auction ID).

## How to adapt

If you have tender data, map columns before calling the pipeline. Example:

```python
import pandas as pd

raw = pd.read_csv("your_tender_events.csv")
renamed = raw.rename(
    columns={
        "bid_timestamp": "Login Timestamp",
        "bidder_id": "User ID",
        "origin_ip": "IP Address",
        "country": "Country",
        "device": "Device Type",
        "user_agent": "Browser Name and Version",
        "bid_valid": "Login Successful",
    }
)
```

Then run `tad quickstart` or `tad run-config` on the mapped file.
