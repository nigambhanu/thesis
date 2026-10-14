ROLE
You are a senior Python/ML engineer doing a careful, reversible code SYNC for my thesis repo
("Eligibility-Aware, Explainable Multi-Bank Credit-Card Recommendation using RAG", Clean Architecture, Python 3.11).
There are two copies of the code and they may differ:
  REFERENCE = my home version, the one that produced my executed results (zip "Thesis-Coding-final.zip", CHANGES.md mentions "round 4").
              Location in this workspace: <<FILL IN: e.g. ./reference_local/ or the folder I pasted>>
  OFFICE    = the repo currently open in this workspace: <<FILL IN: e.g. ./ (workspace root)>>
GOAL: find every difference, decide for each whether it should be ported, get MY approval, apply only what I approve,
then confirm the office repo reproduces the behaviour behind my reference results. A later audit prompt will run after this one.

HARD RULES
- Fully offline. No network, no pip/HF/web. Do not execute project code (no python, pytest, notebooks, scripts).
  Read-only file listing/diff commands (ls, diff, git status/diff/log) are allowed if you have a terminal; otherwise compare by reading.
- NOTHING IS MODIFIED until I reply with an explicit approval token (see Stage 3). Before that you only read and report.
- Never touch data/rule_docs/ or data/eligibility_table.csv, never overwrite results_final/ or any results.json, RESULTS.md, *.jsonl,
  llm_cache.jsonl, artifacts/. These are evidence, not code. If they differ, report only.
- Never delete office-only code. Never "improve" or refactor during sync: sync copies reference behaviour, nothing else.
- Never invent contents. If a reference file is missing, print MISSING. If I did not give you the reference tree, STOP and ask me
  to place it in the workspace or paste the files (order: config/settings.py, src/domain/*, src/pipeline.py, src/model.py,
  src/infrastructure/*, src/application/*, src/api.py, evaluation/*, tests/*, requirements.txt, pyproject.toml, CHANGES.md).
- Tag claims [VERIFIED file:line "quote"] / [INFERRED] / [UNVERIFIED]. One stage per reply; end with "STAGE n DONE - reply NEXT".

KNOWN BEHAVIOUR OF THE REFERENCE (from my executed notebook; use as a fingerprint of "correct")
- CHANGES.md contains the text "round 4" (case-insensitive).
- Dirs: config, data, dashboard, evaluation, src, tests; files: CHANGES.md, EVALUATION_NOTES.md, README.md, pyproject.toml, requirements.txt.
- `python -m src.pipeline` writes artifacts/faiss.index and artifacts/chunks.parquet.
- `python -m evaluation.verify_setup` prints 4 PASS lines: rule docs agree with eligibility table; gold labels equal the oracle;
  index/chunk artefacts match rule docs; grounded/ungrounded prompts built correctly.
- `pytest -q tests` = 33 tests.
- evaluation.run_ablation exposes: load_gold_set, build_retriever(k), make_llm_clients(use_stub=...);
  CLI flags --split {test,dev}, --limit, --out-dir, --arms (A1..A6, bare --arms = core), --no-core, --use-stub-llm, --a4-models.
  Split "test" = every 3rd gold profile; "dev" = the rest. Results merge into results.json. Prints "CAVEAT: ... OPTIMISTIC UPPER BOUND ...".
- evaluation.make_report --results-dir writes RESULTS.md.
- src.model.RAGRecommender(retriever, llm).recommend(profile) returns (decisions, k); per-card prompting, 8 cards for 2 banks.
- src.infrastructure.llm_client.LocalLLMClient(model_name): attributes model_name, max_new_tokens (=1536 in reference);
  methods generate(prompt), generate_batch(list); env: USE_STUB_LLM, LLM_MODEL_NAME, LLM_CACHE_PATH, LLM_CACHE, LLM_BATCH_SIZE.
- Greedy decode (temperature 0.0, do_sample False). Embedder all-MiniLM-L6-v2, 384-d, normalised; FAISS IndexFlatIP in IndexIDMap2.
- API endpoints: POST /recommend, GET /stats, GET /telemetry, GET /history.
Treat any office deviation from these as HIGH priority.

=== STAGE 0: INVENTORY ===
0.1 Print both trees (relative paths, ignoring .git, __pycache__, .venv, node_modules, artifacts/, results_final/, *.pyc).
0.2 Table: path | in REFERENCE | in OFFICE | identical (content hash or byte-compare by reading) | status
    status in {SAME, DIFFERENT, REFERENCE-ONLY, OFFICE-ONLY}.
0.3 Report whether OFFICE is a git repo; if yes show branch, uncommitted changes (git status) and last 5 commits (read-only).
    Recommend creating a backup branch/zip BEFORE I approve changes, and give me the exact commands (do not run them).

=== STAGE 1: CLASSIFY EVERY DIFFERENCE ===
For each DIFFERENT / REFERENCE-ONLY / OFFICE-ONLY file produce an entry:
  ID (D01, D02...) | path | what differs (function/class/line ranges, short unified diff) | category | direction | risk | recommendation
Categories (pick one):
  A  RESULT-AFFECTING LOGIC  (prompts, chunking, retrieval k, decode config, parsing, metrics, split logic, oracle, gold set, McNemar)
  B  PLUMBING / API / UI     (FastAPI, dashboard, config plumbing, logging, Docker, README)
  C  OFFLINE / ENVIRONMENT   (paths, model locations, HF offline flags, proxies, internal package mirrors, GPU/CPU selection)
  D  DEPENDENCIES / TOOLING  (requirements, pyproject, ruff/mypy config, pinned versions)
  E  TESTS
  F  DATA / EVIDENCE         (rule_docs, eligibility_table.csv, gold set, results, caches) - REPORT ONLY, never sync
Direction: REFERENCE-NEWER (office is behind) / OFFICE-NEWER (office has intentional changes) / DIVERGENT (both changed) / UNKNOWN.
Use CHANGES.md, comments, docstrings and any date hints to infer direction; say when it is only a guess.
Recommendation (exactly one): PORT (copy reference into office) / KEEP-OFFICE / MERGE (show the merged result) / ASK-ME / IGNORE.
Rules of thumb:
  - Category C office-specific changes (internal mirrors, local model paths, offline flags) -> default KEEP-OFFICE.
  - Category A where office differs from the fingerprint above -> default PORT, but flag HIGH risk and explain the effect on my reported numbers.
  - Anything where office looks like a bug fix I do not know about -> ASK-ME with your reasoning.
  - Category F -> IGNORE (report only), but tell me if the office data would change the gold labels or chunk count.
Then print three rollups: counts per category, list of HIGH-risk items, list of ASK-ME items.

=== STAGE 2: DOES IT MAKE SENSE? (decision support, still no changes) ===
For every item with recommendation PORT, MERGE or ASK-ME, answer plainly:
  a) Would applying it change any number in my reported results (TEST A1 grounded 0.400 / DEV 0.225, RQ1-RQ4, A2-A6)? Which and why.
  b) Does it make the office repo more or less consistent with the REFERENCE FINGERPRINT?
  c) Could it break an office-only constraint (offline loading, internal paths, security policy)?
  d) Does it need a dependency not already installed in the office environment? (Do not install; just name it.)
  e) Verdict: ACCEPT / REJECT / NEEDS-YOUR-DECISION, with one-sentence reason.
Order the list: category A first, then C, B, D, E.
Finish with a proposed APPLY PLAN: ordered batches (e.g. Batch 1 = items D03, D07 ...), each small enough to review,
and the exact files each batch will write. Then STOP and ask me.

=== STAGE 3: APPROVAL GATE (only after I answer) ===
Accepted replies from me:
  "APPLY ALL ACCEPTED"          -> apply every item you marked ACCEPT.
  "APPLY D03 D07 D12"           -> apply exactly those IDs.
  "REJECT D05" / "KEEP D09"     -> record the decision, do not touch the file.
  "EXPLAIN D04"                 -> show a fuller side-by-side with examples, then ask again.
Anything else = not an approval. Do not infer approval from silence or from "ok" about something else.
When applying:
  1. Work batch by batch, smallest first; after each batch print the full final content of every file you changed (no "..." elisions)
     and a unified diff against the pre-change office version.
  2. Preserve office-only lines marked KEEP-OFFICE; for MERGE items show exactly which lines came from which side.
  3. Preserve all docstrings and type hints; do not reformat unrelated lines.
  4. Do not run anything. Never touch category F or results/evidence files.
  5. Keep an append-only log, printed at the end: ID | file | action | approved by my message | rollback instruction.

=== STAGE 4: POST-SYNC STATIC VERIFICATION ===
Without executing code, confirm the office repo now matches the reference fingerprint:
  - Re-compare the files touched; list what still differs and why (all should be KEEP-OFFICE / category C / F).
  - Walk the import graph of src/ and evaluation/ for the changed modules: every import resolvable, no circular imports,
    domain imports nothing from application/infrastructure, oracle reads only eligibility_table.csv.
  - Check the fingerprint list item by item (flags, function names, defaults, endpoint names, test count by counting `def test_`
    including parametrised cases as best you can; say if it is not exactly 33 and why that may be fine).
  - Check settings: model-name / path variables still point to OFFICE-local locations (do not lose office path overrides).
  - mypy/ruff cleanliness by reading: unused imports, missing annotations on changed code.
Print a table: Check | PASS / FAIL / UNVERIFIED | evidence. Then print "SYNC COMPLETE" or the list of blockers.

=== STAGE 5: HAND-OFF ===
Print: (1) a short change summary for my notes, (2) the exact commands I should run myself, offline, to confirm
(pytest -q tests; python -m evaluation.verify_setup; the stub ablation with --use-stub-llm --split test --limit 6 --out-dir <tmp>) -
as text only, do not run them, (3) the line "Now run the OFFLINE AUDIT prompt." and nothing more.

Start now with STAGE 0.
