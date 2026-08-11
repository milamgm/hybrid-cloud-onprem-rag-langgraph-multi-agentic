# Governance and evidence

`src.governance` records model/prompt/policy versions, policy decisions,
metrics, incidents and risk reviews as hash-chained JSONL evidence. Record only
identifiers, hashes and reason codes: raw prompts, outputs and credentials are
explicitly rejected.

## Azure

Set `APPLICATIONINSIGHTS_CONNECTION_STRING` to export OpenTelemetry telemetry
through Azure Monitor. Configure Microsoft Purview separately to catalog data
assets and lineage; the application deliberately does not upload prompts or
outputs to Purview automatically. A privacy and retention review must approve
any use of Purview `contentActivities`.

## On-premise

Start the OpenTelemetry Collector with `SIEM_OTLP_ENDPOINT` and its
authorization header. It accepts OTLP on localhost ports 4317 and 4318 and
forwards traces, metrics and logs to your SIEM. For a formal GRC workflow,
export approved evidence records into IBM watsonx.governance Factsheets; this
is a governed integration decision, not an unauthenticated webhook.
