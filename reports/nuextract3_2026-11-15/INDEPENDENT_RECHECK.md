# Independent NuExtract3 recheck

15 November 2026. No inference rerun and no changes to raw model records.

- Both tracks contain exactly 512 unique known task IDs and one recorded attempt per case. Suite/task-plan hashes match the frozen files.
- Replayed direct status/value correctness: 466/512 (91.02%). Re-parsing raw JSON and current value normalization reproduces stored values.
- Replayed exact extraction: 426/512 (83.20%). Replayed deterministic calculator correctness: 443/512 (86.52%). Every saved extraction grade and executor result matches current contract replay.
- No token-cap flags. Direct-QA saved evidence quotes all occur verbatim in the corresponding documents; no ANSWER record lacks evidence. This does not prove semantic entailment. Extraction's 512/512 substring-grounded cases include empty fields and are not 512 correct answers.
- Saved prompts contain no serialized expected/reference fields. Source inspection shows inference receives the document/question/schema rather than private reference operands. This is not a general proof against every possible form of leakage.

The original NuExtract3 summary and renderer contained unsupported hardcoded historical scores: NuExtract Tiny 421/416, GLiNER 394/382, Smol 247/272/128. The recorded artifacts instead show Tiny 88/89, GLiNER 214/214, Smol 10/10 and direct QA 1/512. The historical report/figure now read those records directly. NuExtract3 scores were unchanged. The report's Python 3.12 claim was corrected to recorded Python 3.14.6. A count of one-attempt records was renamed so it is not described as successful answers.

NuExtract3's direct QA surpasses Reader-1 overall (466 vs 330), particularly on conflicts, percentages and table lookup. Reader-1 remains stronger on dates (43/43 vs 36/43) and irrelevant instructions (38/43 vs 12/43); both achieve 43/43 corrections. These are parameterized diagnostic tasks rather than population-level accuracy estimates. The calculator track uses human-specified operations and remains separate from direct QA. Strict output-format failures limit interpretation of the earlier Smol comparison.

Remaining provenance limits: the runner declares a model revision but loads the repository name without passing that revision; the sole currently cached snapshot matches the declaration. Weight-byte hashes were not recorded in these run artifacts. Resume logic does not validate model/prompt/settings identity before appending. These are reproducibility weaknesses, not evidence that the saved correctness totals are wrong. Future runs should pin the loaded snapshot, capture weight hashes, and validate resumption compatibility. Hardware/power settings are declared metadata, not independently measured by this recheck.

Machine-readable source fingerprints and verified counts are in independent_recheck.json. Raw inference files remain unchanged.
