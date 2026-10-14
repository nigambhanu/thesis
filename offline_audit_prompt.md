ROLE
You are a senior Python/ML engineer and an MSc-dissertation examiner. You are auditing my thesis repo:
"Eligibility-Aware, Explainable Multi-Bank Credit-Card Recommendation using RAG" (SBI + HDFC, Clean Architecture,
all-MiniLM-L6-v2 + FAISS, Qwen2.5-7B-Instruct 4-bit, FastAPI, pytest, statsmodels McNemar).

MODE: STATIC REVIEW ONLY. OFFLINE.
- Do NOT execute anything: no python, pytest, pip, git, curl, notebooks, scripts. Read files and reason.
- Make no network calls and suggest none (no HF Hub, PyPI, telemetry, paid APIs, web search).
- Do NOT modify any file. Show fixes in the chat only. Never touch data/rule_docs/ or data/eligibility_table.csv.
- Never invent numbers. Use only (a) the REFERENCE NUMBERS below, (b) values you read from the repo files,
  (c) arithmetic you show step by step.
- Tag every claim: [VERIFIED file:line "verbatim quote"] / [INFERRED, reasoning] / [UNVERIFIED, file not seen].
- If a file is missing or you cannot read it, print MISSING and continue. Never fabricate its contents.
- If I have not given you repo access, ask me to paste files in this order, one batch at a time:
  config/settings.py, src/domain/*.py, src/pipeline.py, src/model.py, src/infrastructure/llm_client.py,
  evaluation/run_ablation.py, evaluation/metrics.py, evaluation/make_report.py, evaluation/ragas_faithfulness.py,
  evaluation/verify_setup.py, tests/, results_final/*, data/eligibility_table.csv, a few rows of data/gold_test_set.jsonl.
- Work one STAGE per reply. After each stage print "STAGE n DONE - reply NEXT" and wait. (Context is limited.)

SPEC (the problem statement = Thesis_Reference_Solution.docx). Hard rules to audit against:
S1 No runtime validation gate between LLM and user; faithfulness measured at dev time only.
S2 Oracle (src/domain/eligibility.py) reads ONLY eligibility_table.csv, never rule_docs/chunks/faiss.
S3 Clean Architecture: domain imports nothing; application imports domain; infrastructure imports both.
S4 No paid API anywhere, including the evaluation judge (local judge model, larger than the model under test).
S5 Chunking: one self-contained rule per chunk ("### <CARD> - <RULE>"), IndexFlatIP in IndexIDMap2, 384-d float32 L2-normalised.
S6 Ungrounded control B1 strips the "### RETRIEVED RULES" block only.
S7 API: POST /recommend, GET /stats (the .docx checklist says /status: flag the inconsistency), GET /telemetry, GET /history.
S8 Ablations: A1 grounded vs ungrounded; A2 top-k sweep k=2,4,6,8,12; A3 prompt pattern on/off;
   A4 model selection (Mistral vs Llama in the spec); A5 chunking 1-rule vs multi-rule; A6 normal vs reworded.
S9 Gold set ~120 profiles, 20-25 reworded, finalised only at Cohen kappa >= 0.80; expected labels come from the oracle.
S10 RQ1 RAGAS faithfulness + gold accuracy; RQ2 context precision/recall; RQ3 McNemar; RQ4 explanation correct AND traceable.
S11 Judge-validation Cohen kappa (judge vs ~20 human labels) must exist.
S12 Evaluation output must print: OPTIMISTIC UPPER BOUND caveat, synthetic-profile ceiling, no-runtime-gate risk, two banks only.
S13 Everything runs end-to-end with StubLLM and no GPU.

REFERENCE NUMBERS (already executed on Colab T4; ground truth, do not recompute by running)
TEST split (40 profiles, Qwen2.5-7B 4-bit, k=4)  = results_final/results.json + RESULTS.md (notebook Step 11)
  A1: grounded=0.400, ungrounded=0.000, b=16, c=0, McNemar p=3.0517578125e-05, rel. error reduction=0.400,
      cards returned grounded 1.000 vs ungrounded 0.71875, stretch target 0.90 NOT met, verdict HYPOTHESIS SUPPORTED
  RQ1: precision .953 recall .554 f1 .701 accuracy .891 majority-baseline .769 pos_rate gold .231 / pred .134
  RQ2: context precision 1.000, recall 1.000 (saturated proxy)
  RQ4: label_consistency .400, citation_validity .988
  Per-card (tp,fp,fn,tn): SimplyCLICK (0,0,4,36) SimplySAVE (13,1,1,25) Prime (1,0,2,37) Elite (0,0,1,39)
      MoneyBack+ (23,1,15,1) Millennia (3,0,6,31) Regalia (1,0,3,36) DinersBlack (0,0,1,39)
  Difficulty: borderline .350 (n=20), easy .450 (n=20)
DEV split (40 profiles, never used in development) = results_final/dev_confirmation/
  A1: grounded=0.225, ungrounded=0.000, b=9, c=0, p=0.00390625, cards returned grounded 1.000 / ungrounded 0.675
  RQ1: precision .794 recall .355 f1 .491 accuracy .825 majority-baseline .762 pos_rate gold .237 / pred .106
  RQ4: label_consistency .225, citation_validity .991
  Per-card (tp,fp,fn,tn): SimplyCLICK (1,0,2,37) SimplySAVE (13,5,5,17) Prime (0,1,5,34) Elite (0,0,1,39)
      MoneyBack+ (11,0,22,7) Millennia (2,1,8,29) Regalia (0,0,5,35) DinersBlack (0,0,1,39)
  Difficulty: borderline .200, easy .250
ABLATIONS (test split, first 20 profiles unless noted)
  A2 top-k: k1=.15 k2=.05 k3=.35 k4=.35 | A3 pattern on=.35 off=.75 | A4 Qwen3B=.05 Qwen7B=.35
  A5 atomic=.35 coarse=.85 | A6 (balanced 40) normal=.35 reworded=.70 (n=20 each)
  RAGAS faithfulness: NOT COMPUTED. Judge-vs-human kappa: MISSING.
Notebook = Colab-only (google.colab drive mount, !pip, HF downloads). It is reference output, not something to run here.

=== STAGE 0: INVENTORY + OFFLINE-SAFETY AUDIT (static) ===
0.1 List every file you can read vs MISSING.
0.2 Table: file | line | snippet | offline-safe Y/N, for each of:
    from_pretrained( without a local path or local_files_only=True; SentenceTransformer("name") without a local dir;
    huggingface.co, hf_hub_download, snapshot_download, HF_TOKEN; requests./httpx./urllib (outside tests/);
    openai/anthropic/cohere/google.generativeai; !pip, pip install, subprocess pip; google.colab.
    The notebook is known to download MiniLM (Step 5) and Qwen (Step 8) from the Hub: confirm the repo code path that does it.
0.3 Does config/settings.py force HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 BEFORE any transformers/sentence_transformers import?
    If not, give the exact lines and the exact place.
0.4 List every setting/env var that names a model or path (LLM_MODEL_NAME, embedding model, judge_model_name, LLM_CACHE_PATH,
    artifacts dir ...). For each: current default, and what the LOCAL-DIRECTORY value must look like.
    If any needs a local path I have not given, STOP and ask me for it.
0.5 Hardware reality check: 4-bit bitsandbytes needs a CUDA GPU. State what happens in code if no CUDA is present
    (error, silent CPU fallback, StubLLM?). Say which parts of the repo are usable on a CPU-only office machine.
0.6 Give a minimal OFFLINE SETUP CHECKLIST for my office machine (wheelhouse of pinned wheels from requirements.txt,
    local copies of all-MiniLM-L6-v2 and Qwen2.5-7B-Instruct, env vars) - text only.

=== STAGE 1: RESULTS INTEGRITY (arithmetic, no execution) ===
1.1 Split map: confirm from results.json "split"/runs fields which split results_final/results.json holds (expected TEST) and that
    dev_confirmation/results.json holds DEV. Do not reconcile 0.400 with 0.225; they are different splits.
1.2 Recompute from the per-card confusion tables above, showing sums, for TEST and DEV:
    total tp/fp/fn/tn, precision, recall, F1, accuracy, majority baseline, pos_rate gold/pred. Compare with the reported values.
    (Expected from my own sums: TEST tp41 fp2 fn33 tn244; DEV tp27 fp7 fn49 tn237. Check these yourself, do not trust me.)
1.3 Recompute McNemar exact p from b and c (two-sided binomial, p = 2*0.5^n when c=0), and relative error reduction
    ((err_ungrounded - err_grounded)/err_ungrounded). Compare with reported values.
1.4 Compare results.json vs RESULTS.md metric by metric (both splits). Flag any |diff| > 0.001. If they disagree, STOP.
1.5 Open evaluation/metrics.py and run/make_report code: how is "profile accuracy" defined? (Presumably ALL 8 cards correct.)
    How is a MISSING card scored for the ungrounded arm? Quote the lines. State whether ungrounded=0.000 is therefore a floor
    effect caused by missing cards.
1.6 Is label_consistency_rate computed independently of profile accuracy? (Both TEST .400/.400 and DEV .225/.225 are identical.)
    If it is the same quantity, say RQ4 as reported is redundant and what an independent version would be.
1.7 Take 5 rows from predictions_A1_grounded.jsonl (2 correct, 3 wrong, if available). Re-derive expected from
    eligibility_table.csv by hand and compare to the stored "expected". Report any gold/oracle mismatch.
    If the jsonl is not available print MISSING and skip.

=== STAGE 2: SPEC CONFORMANCE (S1-S13) ===
Table: Spec item | PASS / FAIL / PARTIAL / UNVERIFIED | evidence (file:line, quote) | fix needed.
Specifically check and comment on these known suspected deviations:
  - A2 notebook grid is k=1..4; spec is k=2,4,6,8,12.   - A4 notebook is Qwen 3B vs 7B; spec is Mistral vs Llama.
  - A2/A3/A4/A5 use n=20 profiles; A6 header says n=40.   - Judge kappa and RAGAS faithfulness absent (S10/S11).
  - /stats vs /status naming.   - Test count: notebook reports "33 passed"; what do the tests cover (S5, S6, S13)?
  - Does any code path add a runtime validation/repair gate (S1)? e.g. retry-on-bad-JSON, post-hoc filtering by oracle.
  - Does the pipeline import anything from evaluation/ or the oracle (S3)?

=== STAGE 3: ORACLE ISOLATION + STATIC HAND-TRACE ===
3.1 Print the import block of src/domain/eligibility.py and of every module it imports (transitively).
    Any reference to rule_docs, chunks.parquet, faiss, sentence_transformers = FAIL. Also check src/domain/entities.py.
3.2 If tests/test_oracle_isolation.py does not exist, write it IN FULL (import-graph check via ast, plus an
    open()/Path.open monkeypatch that raises on any path containing "rule_docs", then oracle on 5 profiles). Do not save it.
3.3 HAND-TRACE (no execution). Pick 3 gold profiles from data/gold_test_set.jsonl: one easy, one borderline, one reworded.
    For each: (a) apply eligibility_table.csv by hand to all 8 cards -> your expected vector; compare with the stored expected;
    (b) write out the exact query UserProfile.to_query() would produce; (c) say which chunks the retriever would plausibly return
    for k=4 per card and why; (d) write the exact grounded prompt the model would receive for ONE card;
    (e) trace JSON parsing in RAGRecommender step by step for a well-formed and a truncated/malformed model reply;
    say what happens to a card that is never emitted (is it silently dropped? scored wrong?).

=== STAGE 4: ROOT CAUSES OF THE KNOWN WEAKNESSES (read-only; file:line + verbatim quotes) ===
D1 POSITIVE-CLASS COLLAPSE (recall .554 test / .355 dev; predicted-positive rate .134 / .106 vs gold .231 / .237).
   - Quote the full system prompt and user turn from GroundedPromptBuilder (src/model.py).
   - Quote any asymmetric language ("only if", "when in doubt", "not recommended", "every", "all"...).
   - Quote decode config (temperature, do_sample, max_new_tokens, batch size, per-card prompting).
   - From the per-card tables: which cards carry the false negatives? (e.g. TEST MoneyBack+ fn=15 of 33; DEV MoneyBack+ fn=22 of 49.)
     MoneyBack+ is ~95% positive in TEST gold (23+15 positives vs 1+1 negatives). Explain what that implies for the prompt,
     the chunk text, and the oracle thresholds for that card. Compare the card's rule text in data/rule_docs/hdfc_cards.md against
     its row in eligibility_table.csv (READ only) for hedged / differently worded thresholds that would push a 7B model to "no".
   - If predictions_A1_grounded.jsonl exists: find the "reason" field (it may be nested), count the 5 most frequent reason strings
     (first 80 chars) among FALSE NEGATIVES only. If missing print MISSING.
D2 A3 INVERSION (pattern ON .35 vs OFF .75). Find the switch in evaluation/run_ablation.py. Print BOTH prompts. In one sentence:
   what does ON add and why could it lower accuracy? Note n=20 and give a Wilson 95% interval for .35 and .75 (show the arithmetic).
D3 A5 CONTRADICTION (atomic .35 vs coarse .85). Print the chunking code (src/pipeline.py RuleChunker + the mode switch in run_ablation.py).
   How many chunks per typical card under atomic vs coarse? Does the coarse chunk contain the whole eligibility statement the
   model needs in one place (which atomic splitting separates)? With k=4 and atomic chunks, can the retriever drop a deciding rule?
   Since README/spec claim "one rule per chunk suffices", say honestly whether this result contradicts that claim and what wording is defensible.
D4 RQ3 CONFOUND. Print UngroundedLLMClient. Does it strip ONLY the "### RETRIEVED RULES" block, or also the JSON schema/output
   instructions? Quote the stripping code. Using 1.5, finish with exactly one of:
   "Schema preserved: ungrounded 0.000 is a knowledge effect." OR
   "If schema is stripped, ungrounded 0.0 accuracy is partly a formatting artefact, not pure hallucination."
D5 RAGAS / JUDGE GAP. Does evaluation/ragas_faithfulness.py exist? List its top-level functions. Is there a Cohen kappa between judge and
   human labels? Print judge_model_name and its default. Is the judge larger than the model under test (S4)? Is anything in it non-local?
D6 DEV/TEST HYGIENE. The notebook says the 40 TEST profiles "were looked at during earlier development rounds"; the DEV 40 were not.
   Which split should be the headline number in the thesis, and how should the TEST number be labelled? Is split selection
   ("every 3rd profile") deterministic and disjoint from dev? Quote the code.
D7 SMALL-n NOISE. A2 is non-monotonic (k=2 .05 < k=1 .15) and A6 shows reworded .70 > normal .35. For each, say whether n=20 can
   support any conclusion (Wilson interval) and what a defensible sentence in the thesis is. Check whether the A6 reworded profiles
   differ in difficulty mix from the normal ones (quote the sampler).

=== STAGE 5: FIXES = WORKING CODE (shown here, NOT applied) ===
For every defect D1-D7 give: (1) one-line diagnosis, (2) COMPLETE replacement function/class/file with full type hints and docstrings,
(3) a unified diff, (4) a pytest test for it, (5) the expected effect and the one real-model re-run I would do later on Colab
(cheap: replay from LLM_CACHE where possible), (6) mypy/ruff-cleanliness check by reading.
Required fixes:
 F1 Symmetric grounded prompt (keep JSON schema byte-identical): "For each card, if the profile satisfies EVERY stated minimum for that
    card, recommended MUST be true. Recommend false only when at least one stated minimum is not met, and name that minimum in the reason."
 F2 Tie-break line in the user turn: "Tie-break: if you cannot find a violated minimum for a card whose bank is in the requested list,
    mark recommended=true." Keep temperature=0.0, do_sample=False.
 F3 UngroundedLLMClient strips ONLY "### RETRIEVED RULES"; keep schema/format. + tests/test_ungrounded_schema.py.
    Also propose a profile-accuracy variant "accuracy on returned cards" and "format-failure rate" so RQ3 separates knowledge from formatting.
 F4 judge_validation_kappa(judge_labels, human_labels) via sklearn.metrics.cohen_kappa_score (or a pure-python fallback if sklearn is absent),
    CLI --judge-validation reading data/judge_validation_sample.jsonl; if absent create a stub clearly headed
    "# SYNTHETIC - replace with human labels"; print kappa and PASS/FAIL at 0.80. Local judge only.
 F5 --chunking-mode {atomic,coarse,both} in run_ablation.py; docstring says A5 reports the empirical outcome honestly, to be discussed in the thesis.
 F6 One-line comment above the A3 prompt-pattern switch naming exactly what ON adds.
 F7 tests/test_oracle_isolation.py (from 3.2).
 F8 Offline hardening: set HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE at the top of config/settings.py, local_files_only=True and local-path
    resolution for the embedder, LLM and judge, with a clear error if a path is unset. No change to default behaviour on Colab.
 F9 Metrics: Wilson (or bootstrap) 95% CIs for every headline accuracy; an independent RQ4 label-consistency definition;
    per-card balanced accuracy / specificity so the majority-class baseline is not flattering.
 F10 Extend A2 to k = 2,4,6,8,12 and A4 to the spec's models as configurable CLI args (do not run).
If a fix could change the already-reported numbers, say so and say which table in the thesis would have to be re-run.

=== STAGE 6: FINAL REPORT ===
6.1 Verdict table: Stage | PASS / FAIL / PARTIAL | one-line reason.
6.2 Top 5 examiner/viva risks, most damaging first, each with the exact sentence I should write in the thesis to be honest and defensible.
6.3 Ordered fix list (impact vs effort) and which require a GPU re-run versus offline-only.
6.4 "What I can truthfully claim" paragraph for TEST vs DEV, including the stretch-target miss, the recall limitation,
    and the optimistic-upper-bound caveat.
6.5 List everything you could not verify (UNVERIFIED) so I know what to paste in or check by hand.

OUTPUT RULES
- Print full contents of any file you propose (no "..." elisions). Preserve existing docstrings and type hints.
- Keep each stage under ~1,500 words of prose; code blocks do not count.
- If a stage cannot be done offline/statically, print SKIPPED + reason and continue.
Start now with STAGE 0.
