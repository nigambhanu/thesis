ROLE
You are a senior Python/ML engineer implementing the findings of an OFFLINE AUDIT of my thesis repo
("Eligibility-Aware, Explainable Multi-Bank Credit-Card Recommender using RAG", Clean Architecture, Python 3.11,
Qwen2.5-7B 4-bit, MiniLM + FAISS, FastAPI, pytest). The audit (STAGE 0-6) is finished and produced fixes F1-F10 as code in the
earlier conversation. Your job: incorporate them safely, in reviewed batches, WITHOUT changing the behaviour that produced my
already-reported results unless I opt in.

HARD RULES
- Fully offline: no network, no pip/HF/web search. Do not run project code, tests, notebooks or models. Read and edit files only.
  I will run commands myself and paste output back (see PROTOCOL).
- NEVER edit or delete: data/rule_docs/*, data/eligibility_table.csv, data/gold_test_set.jsonl, results_final/** (results.json, RESULTS.md,
  predictions_*.jsonl, dev_confirmation/**), llm_cache.jsonl, artifacts/**. They are evidence. New outputs go to NEW paths
  (e.g. results_rescored/).
- REPRODUCIBILITY INVARIANT: with default flags the code must behave exactly as the reference did (same prompts byte-for-byte, same
  decode config, same chunking, same k, same split rule). All new behaviour sits behind new flags / new arm names / a prompt-version
  switch whose default is the old one. Prompt text feeds the LLM cache key: do not alter old prompt strings (even whitespace).
- Do not refactor beyond what a finding needs. Keep every existing docstring and type hint. New code: full type hints + docstrings,
  mypy-clean and ruff-clean by reading.
- Keep CHANGES.md containing the text "round 4" (my notebook asserts it). Append a "round 5" section; never remove round 4.
- Never invent numbers. Numbers come from files, from my REFERENCE NUMBERS, or from arithmetic you show.
- Tag claims [VERIFIED file:line "quote"] / [INFERRED] / [UNVERIFIED]. If something is missing print MISSING and ask.
- One stage per reply; end with "STAGE n DONE - reply NEXT". Nothing in a batch is applied until I reply "APPLY BATCH <letter>".

PRECONDITIONS (STAGE 0 - check before any edit; if any fails, STOP and ask)
P1 The office repo has been synced with my reference code (the SYNC prompt finished), or I tell you to proceed anyway.
P2 A backup exists (git branch/tag or zip). If it is a git repo, tell me the commands to create one (do not run them).
P3 You can see the audit's F1-F10 code from the earlier conversation. If not (new chat), tell me to paste the audit's Stage 5 output;
   otherwise re-derive each fix from the FINDINGS below and show it for approval.
P4 DATA CHECK: office data/rule_docs and data/gold_test_set.jsonl must be the same as those that produced results_final.
   Compare them with the reference copy if present (read-only). If they differ, STOP: any rebuild would no longer match my results.
P5 Locate: llm_cache.jsonl, results_final/predictions_A1_*.jsonl (both splits), and the existing per-profile fields. Print where each is
   and which keys a prediction row has. Missing files = MISSING.

FINDINGS TO INCORPORATE (from the audit; re-verify each in code before acting)
Verdicts: S0 offline FAIL; S1 results integrity PASS with 2 flags; S2 spec PARTIAL (5 pass, 6 partial, 2 fail); S3 oracle PASS; S4 root causes
PARTIAL; S5 fixes ready.
 N1 Offline: no HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE; src/pipeline.py ignores settings model_name; hash-embedding fallback is SILENT; every
    from_pretrained points at the Hub.
 N2 RQ4 label_consistency_rate == profile accuracy (TEST .400/.400, DEV .225/.225): not an independent measure.
 N3 A6 is labelled TEST but mixes splits.
 N4 RQ1: in the metrics, a MISSING card is counted as a correct "no". Harmless for grounded (cards returned 1.000) but wrong for any arm
    with missing cards. Report missing separately; never score a missing card as correct.
 N5 RQ3: ungrounded 0.000 is mostly a floor effect: 33 of 40 TEST profiles lacked >=1 card; only 7 were complete and all 7 wrong. The B1
    prompt also swaps the system prompt (goes beyond "strip only the rules", spec S6). Need: accuracy on complete profiles, format-failure
    rate, and a "rules-only" B1 that strips only "### RETRIEVED RULES".
 N6 S9 gold set was never finalised through the kappa gate (independent-labels file missing). S11 judge-validation kappa never ran.
    RAGAS "faithfulness" is actually named/implemented as `fact_score`: naming/claim mismatch.
 N7 /status vs /stats naming; A2 grid is k=1..4 (spec 2,4,6,8,12); A4 is Qwen 3B vs 7B (spec Mistral vs Llama).
 N8 TEST split was inspected during development; DEV (0.225) is the clean headline. TEST 0.400 = "development" number.
    Wilson 95% CIs: TEST [0.264,0.554], DEV [0.123,0.375] (verify arithmetic yourself).
 N9 A5 atomic .35 vs coarse .85 and A3 pattern-off .75 vs on .35 contradict design claims; n=20 each; README/thesis claim "atomic chunks
    suffice" must be reworded: atomic is kept for citation granularity, not accuracy. All rules of a card are retrieved at k=4 in both arms.
 N10 Dominant error = false negatives from misreading lakh-grouped amounts ("2,50,000") and "as young as 18" read as an upper bound.
     All 15 TEST MoneyBack+ FNs show the amount misread. Per-card accuracy beats always-"no" by only .122 TEST / .063 DEV.
 N11 Not verifiable offline: A3/A5 discordant counts (per-profile correctness not stored), A6 difficulty mix, reasons of non-MoneyBack+ FNs,
     why the ungrounded arm dropped cards (raw generations not stored).

FIX BATCHES (in this order; each batch = plan -> diff + full file contents -> my APPLY)

BATCH A - offline-only, no change to any reported number
 A1 F9 `evaluation/rescore_predictions.py` (new): reads results_final/predictions_A1_{grounded,ungrounded}.jsonl for TEST and DEV,
    writes ONLY to results_rescored/. Outputs: Wilson 95% CIs for profile accuracy; format-failure rate; accuracy on complete profiles;
    independent RQ4 (label match AND cited chunk valid for the right card, defined in the file docstring); per-card balanced accuracy and
    specificity; missing cards reported separately. Pure python/pandas/numpy (no sklearn requirement). Mark the file offline-safe.
 A2 F8 offline hardening: set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 at the very top of config/settings.py (os.environ.setdefault,
    before any transformers/sentence_transformers import); embedder, LLM client and judge load from a LOCAL path with local_files_only=True
    when the offline flag is set; src/pipeline.py uses settings.model_name; replace the silent hash-embedding fallback with a hard error
    unless an explicit ALLOW_HASH_EMBEDDING=1 test flag is set; clear message naming the env var when a path is unset.
    Default on Colab (online) must be unchanged.
 A3 F7 tests/test_oracle_isolation.py and tests/test_ungrounded_schema.py (the latter xfail/skip-safe until Batch C lands).
 A4 Metrics: N4 fix (missing card never counted correct) behind a `count_missing_as_wrong: bool = False` parameter so default numbers are
    unchanged; make_report prints both variants. N2: add `independent_label_consistency` next to the old one; keep the old key.
 A5 Labelling: make_report/RESULTS text relabels TEST as "development split (inspected during development)" and DEV as "held-out
    confirmation (headline)"; A6 labelled "mixed split" (N3). Keys in results.json stay the same.
 A6 Docs: README.md + EVALUATION_NOTES.md rewording (N5, N6, N7, N9, N8) using the honest sentences below; rename the RAGAS claim to
    "FactScore-style atomic precision (fact_score), not RAGAS faithfulness"; add /status as an alias of /stats in src/api.py without
    removing /stats; append "round 5" to CHANGES.md.

BATCH B - CLI/config additions, defaults unchanged
 B1 F5 `--chunking-mode {atomic,coarse,both}` in run_ablation.py with the honest-finding docstring (default = what A5 does today).
 B2 F6 one-line comment above the A3 pattern switch naming exactly what ON adds.
 B3 F10 A2 `--topk-grid` default "1,2,3,4" (unchanged), spec grid selectable "2,4,6,8,12"; A4 `--a4-models` already exists: document the
    spec's Mistral/Llama as local-path arguments. Do not run.
 B4 Save per-profile correctness for A3/A5/A6 arms to results.json when available (new keys only) so paired McNemar for A3/A5 can be computed
    later from the cache with the OLD prompts, BEFORE any prompt change. Add `--paired-from-cache` helper only if it needs no model load.

BATCH C - result-affecting, strictly opt-in (new flags/arms; old behaviour byte-identical by default)
 C1 F1/F2/F1b prompt variants as `--prompt-version {v1,v2}`, default v1 = existing text unchanged. v2 contains the symmetric
    "EVERY stated minimum -> recommended MUST be true" rule, the tie-break line, and (F1b as defined in the audit; if unavailable propose it:
    consistent plain-integer rendering of amounts and an explicit "minimum age 18 means age >= 18" cue, done in PROMPT RENDERING only,
    never by editing rule_docs). Same JSON schema byte-for-byte. temperature 0.0, do_sample False.
 C2 F3 new arm `B1_rules_only` (strips only "### RETRIEVED RULES", keeps system prompt and schema); keep the existing B1 unchanged.
 C3 F4 `judge_validation_kappa` + `--judge-validation`; reads data/judge_validation_sample.jsonl; if absent create a stub whose first line is
    "# SYNTHETIC - replace with human labels" and make the CLI print "NOT A RESULT: synthetic labels" instead of PASS/FAIL while the marker is present.
    Local judge only; verify the judge model differs from, and is larger than, the model under test.
 C4 Evaluation hygiene for v2: a NEW split `dev2` (disjoint from test and dev, deterministic rule, documented) so prompt changes are not tuned on
    the DEV headline. Show the split code; if the gold set has too few unused profiles, say how many remain and stop for my decision.

BATCH D - hand-off only (no code edits): a checklist for Colab GPU runs and human tasks, in order, each with exact commands:
 D1 rescore on existing predictions (offline now); D2 paired A3/A5 McNemar from cache with old prompts; D3 hand-label 20 explanations then
 `--judge-validation` with a local judge (GPU); D4 second annotator on a gold subset to close S9; D5 B1_rules_only run (GPU);
 D6 v2 prompt on dev2 (GPU), then compare with v1 on dev2; D7 A4 with spec models if local gated weights exist.
 State for each: GPU needed?, expected runtime from the notebook timings (grounded ~9-14 s/profile after load, B1 ~30-60 s/profile), what
 result file it writes, and which thesis table it updates.

STAGES
 STAGE 0  Preconditions P1-P5. Print table: check | PASS/FAIL/MISSING | evidence. Stop on FAIL.
 STAGE 1  For each batch item: exact files to create/modify, risk to the reproducibility invariant (none/low/needs-flag), and whether it can change
          any reported number. Print the plan, then ask me which batches to apply. Do not edit.
 STAGE 2  On "APPLY BATCH A": apply item by item. After each item print the unified diff and the FULL final content of every new/changed file
          (no "..." elisions). Then run a STATIC verification: imports resolve, no circular imports, no new network calls, defaults unchanged
          (quote the default lines), docstrings/type hints preserved, "round 4" still in CHANGES.md.
          Then give me the commands to run myself, offline: `python -m pytest -q tests`, `python -m evaluation.verify_setup`,
          `python -m evaluation.rescore_predictions --out-dir results_rescored` (read-only on results_final). Wait for my pasted output.
 STAGE 3  Interpret my pasted output: expected test count = 33 + new tests; any failure -> root cause by file:line -> minimal patch -> show -> wait.
          Verify the rescoring numbers against my REFERENCE NUMBERS (TEST A1 0.400 -> 16/40; DEV 0.225 -> 9/40; per-card tables) and report any mismatch.
 STAGE 4  Batch B, same procedure as Stage 2/3 (static check + commands for me; stub ablation: `python -m evaluation.run_ablation --use-stub-llm
          --split test --limit 6 --arms A1 A3 A5 --out-dir <tmp>` to confirm all arms complete without exception; stub numbers are not results).
 STAGE 5  Batch C, only after I explicitly say so. Same procedure. Re-prove the invariant: show that with default flags the v1 prompt strings and decode
          config are identical to the reference (quote both sides).
 STAGE 6  Batch D checklist + final change log table (ID | file | what changed | default-behaviour-changed? Y/N | approved by | rollback) +
          the "what I can truthfully claim" text below, updated for what is now done. Then print "INCORPORATION COMPLETE" or the blockers.

PROTOCOL FOR RUNS
You never execute code. I run the commands and paste the full output. If output shows an import error, an offline-load error, or a failing test,
treat it as the next task; do not guess past it. If a command would need the network or a GPU, say so and move it to Batch D.

HONEST WORDING TO USE (edit only if the files show something different)
- Headline: "On a synthetic, oracle-labelled gold set (SBI, HDFC; 8 cards) the grounded system was fully correct for 22.5% of 40 held-out
  applicants (95% CI 12-38%); on the 40 profiles inspected during development it reached 40.0% (26-55%). Neither approaches the 90% stretch
  target; figures are an optimistic upper bound."
- RQ3: "The ungrounded control returned all eight cards for only 7 of 40 profiles, so its 0% combines output-format failure with wrong answers
  (0 of 7 complete profiles correct); grounding improves end-to-end correctness, not hallucination in isolation." Replace with the B1_rules_only
  numbers once they exist.
- Chunking: "Atomic chunking is retained for citation granularity, not accuracy; coarse chunks scored higher (0.85 vs 0.35, n=20, CIs overlap only partly)."
- Errors: "Errors are dominated by false negatives from misread lakh-grouped amounts and minimum-age phrasing; recall is the informative measure,
  per-card accuracy barely exceeds the always-reject baseline."
- RQ4/RQ1 faithfulness: "No judge-validation kappa or RAGAS faithfulness was computed; explanation quality is evidenced only by citation validity (98.8% / 99.1%)."

OUTPUT RULES
Prose per stage under ~1,200 words (code blocks excluded). If a step cannot be done offline/statically print SKIPPED + reason and continue.
Start now with STAGE 0.
