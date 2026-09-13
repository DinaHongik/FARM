# Semantic binding evaluation protocol

The 150 requests and 450 completed configuration predictions are frozen before reference preparation. This evaluation is conditional on supplied correct endpoints. There is no new training, workflow generation, account lookup, or platform execution.

## Reference preparation and blinding

Two separate model families propose reference decisions independently from the same request, frozen endpoint metadata, and archived official endpoint documentation. Neither sees workflow predictions, arm names, checker scores, or the other proposal. Model agreement is agreement between automated annotators, not human gold or independent proof of truth. The same generation backbone is not used as a reference annotator.

Official documentation is collected by endpoint URL only. Applet examples, unrelated user recipes, advertisements, and recommendations are excluded. Current documentation may differ from the frozen experiment. Such differences are recorded; new account information or current values must never be silently treated as supplied inference evidence. Input-field control descriptions can clarify meaning, but do not retroactively give models knowledge they were not supplied.

Each field reference states whether its admissible decisions can be enumerated, requires missing context, is open-ended, or is unjudgeable. Closed references list all known permissible source/value alternatives with evidence quotes. Open-ended free-text generation must not be judged incorrect merely because it differs from a short list of model-written examples. Unknown field types, unavailable enum IDs, omitted account resources, and ambiguous requests remain explicit. Trigger setup cannot use the output of that future trigger.

Human review must be recorded explicitly, with reviewer name, date, evidence, and any changes. A human-approved decision is not called independently double-annotated unless a second actual human review is recorded. Software never fills these fields automatically. Until review, every reference and score is labeled AI-assisted and provisional. Disagreements and fields with invalid references remain unresolved; they are not silently converted to correct answers.

## Frozen scoring rules

- A supported dynamic binding names a declared selected-trigger ingredient and the correct action field. A static binding must reproduce an allowed normalized value from an exact request span under a declared transformation. Merely matching the field type is insufficient.
- A required missing value and an unnecessarily omitted available value are different outcomes. Scoring the `needs_input` decision measures missing-information identification, not the semantic quality of the question text.
- Optional omission is permitted only when frozen requiredness is explicitly false and the reference permits omission. Missing requiredness is not treated as optional.
- Complete coherent alternative binding sets are compared as alternatives. Do not union mutually exclusive alternatives into a larger target set. Open-ended or unresolved targets are not scored as exact mismatches.
- Malformed, duplicate, missing, and extra predictions remain model failures. Do not drop failed requests from the 150-case population.

Report (1) exact field-decision accuracy on independently scorable fields, with both micro and macro values; (2) ingredient-to-field precision/recall/F1 on cases with a reviewed closed binding reference; (3) missing-information decision accuracy; (4) whole-configuration decision correctness only for fully adjudicated cases, distinguishing correctly unresolved cases from fully bound configurations; and (5) initial-to-final repair gains and regressions. Empty binding sets do not receive an artificial perfect F1. Report all relevant counts, judgeable coverage, reference disagreement, and unjudgeable cases. Confidence intervals describe the fixed predictions and references, not uncertainty in an automated judge's correctness.

AI-assisted pilot scores and human-reviewed scores are separate artifacts. The scorer must refuse to label an incomplete or model-only reference set as human gold. Neither schema conformity nor semantic-reference agreement is platform execution success.
