# RecruitIQ LLM Rollout Guide

This project supports three scoring modes via `SCORING_MODE`:

- `legacy`: algorithmic scoring only (no parallel LLM scoring)
- `hybrid`: blend LLM and algorithmic for semantic dimensions (recommended default)
- `llm_first`: prefer LLM semantic scores with algorithmic fallback per dimension

## Recommended rollout sequence

1. Start in `hybrid` mode in production.
2. Track quality, latency, and failure rate for at least one evaluation window.
3. Compare against baseline `legacy` results.
4. Move to `llm_first` only if stability and quality targets are met.

## Suggested acceptance gates

- Quality: no statistically significant drop in hire-signal quality vs `legacy`
- Reliability: no increase in endpoint failure rate
- Latency: p95 screening latency remains within agreed SLO
- Fallback health: warning/fallback events trend down and remain bounded

## Rollback plan

Rollback is configuration-only:

1. Set `SCORING_MODE=legacy`
2. Restart API service
3. Re-run health check and one representative screen request

No schema rollback is required for mode rollback.

## Operational notes

- `warning_flags` are persisted for internal diagnostics.
- `llm_provider`, `llm_model`, and `prompt_version` are stored with score results for auditability.
- Public API response remains stable while internal warning metadata is retained in storage.
