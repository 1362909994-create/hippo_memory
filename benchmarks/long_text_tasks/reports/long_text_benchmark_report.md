# Long Text Benchmark Report

Run: `20260623T203000Z`
Mode: `local`

| Group | Task | A score | B score | Delta | B pass | Rule hit | Compression | Leaks |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| G1_hidden_long_rule_task | G1_smoke_hidden_long_rules | 4.67 | 9.90 | +5.23 | 1.00 | 1.00 | 0.90 | 0 |
| G2_needle_in_haystack | G2_needle_100k | 4.67 | 10.00 | +5.33 | 1.00 | 1.00 | 1.00 | 0 |
| G2_needle_in_haystack | G2_needle_10k | 4.67 | 9.99 | +5.32 | 1.00 | 1.00 | 1.00 | 0 |
| G2_needle_in_haystack | G2_needle_50k | 4.67 | 10.00 | +5.33 | 1.00 | 1.00 | 1.00 | 0 |
| G3_version_conflict | G3_version_conflict | 4.00 | 9.63 | +5.63 | 1.00 | 1.00 | 0.63 | 0 |
| G4_noise_suppression | G4_noise_suppression | 4.00 | 9.69 | +5.69 | 1.00 | 1.00 | 0.69 | 0 |
| G5_compression_fidelity | G5_compression_fidelity | 4.00 | 9.94 | +5.94 | 1.00 | 1.00 | 0.94 | 0 |
| G6_privacy_sensitive | G6_privacy_sensitive | 4.00 | 9.21 | +5.21 | 1.00 | 1.00 | 0.21 | 0 |

## Analysis

This matrix checks whether hippo_memory helps on long-document tasks without leaking hidden answers to the baseline prompt. It covers hidden long rules, needle retrieval, version conflict handling, noise suppression, compression fidelity, and privacy-sensitive memory filtering.

## Recommendations

- Treat local mode as deterministic harness validation, not a Codex quality claim.
- Use codex mode only after local mode passes, because it calls the external model API.
- Keep hidden answer artifacts out of condition prompts and only write them after both runs.
