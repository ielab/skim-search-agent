# Release checklist — before the anonymous GitHub push

Prepared 2026-08-01. Nothing here is committed or pushed; every item is a decision or a scrub the
author performs at release time.

## 1. Anonymity scrubs (blocking)

- [x] Cluster account + absolute paths scrubbed (2026-08-02); HF handle in corpus_build kept (public repo is de-anonymized).
  former cluster account and the HF handle). Historical note; scrub done 2026-08-02:
  `grep -rln 'wan458\|OD-236007\|wshuai190' scripts/ configs/ corpus_build/ | grep -v __pycache__`
  Replace the SLURM account with a `$SLURM_ACCOUNT` env default and absolute paths with
  repo-relative ones.
- [ ] **HF dataset links**: README.md and corpus_build/README.md currently use the `<HF_ORG>`
  placeholder / the real handle respectively. Either (a) mirror the datasets to an anonymous HF
  org and fill `<HF_ORG>`, or (b) note "released on acceptance". Remember
  corpus_build/README.md still carries the real handle and the "private on purpose" note.
- [ ] **De-obfuscation tension**: corpus_build/README.md warns the BrowseComp-Plus twin
  de-obfuscates a benchmark corpus and must not be fully public. Decide the gating story
  (gated HF repo + request-access) and make README wording consistent with it.
- [x] `Boolean_agent_paper/` removed from the release (author decision 2026-08-01). If ever re-added, check
  `\author`, acknowledgments, and any grant numbers in the LaTeX.
- [ ] `logs/`, `slurm_logs/`, `runs/`, `docs/run_manifest.md`, `docs/paper_loop_log.md`,
  `docs/reviews/`, `CURRENT.md`, `NEXT_STEPS.md`, `CLAUDE.md`, `comparison_result.md`,
  `_audit_tmp/` contain internal history with identifying paths — exclude via `.gitignore`
  (below) or prune before push.

## 2. Suggested .gitignore for the public repo

```
envs/
data/
indexes/
runs/
logs/
slurm_logs/
_audit_tmp/
__pycache__/
*.premerge.bak
```

(`runs/` is ~large and contains full trajectories with model outputs; release a small sample or
metrics-only export if reviewers should see raw episodes.)

## 3. Runnability checks (do on a clean machine)

- [ ] `pip install -r requirements.txt` on a fresh Python 3.10 venv; `pytest tests/` green.
- [ ] `scripts/build_indexes.sh` runs with only `data/` populated (no cluster modules assumed).
- [ ] The Quickstart cell in README.md (limit 20) completes against a local vLLM endpoint.
- [ ] `analysis/make_paper_tables.py --check` degrades gracefully when `runs/` is absent
  (currently it assumes the full run tree; consider a `--demo` note in the README instead).
- [ ] `scripts/shard_cell.sh` SBATCH headers: partition `h24gpu`, the account, and the
  `EXCLUDE_NODES=g047` default are cluster-specific — parameterize or document as such.

## 4. Documentation state (done in this pass)

- [x] Top-level `README.md` rewritten as the SIEVE front page (figures in `docs/assets/`).
- [x] `docs/REPRODUCING.md` — end-to-end pipeline (env → data → indexes → serving → cells →
  sharding → recovery → judging → tables/figures) + configuration reference.
- [x] `corpus_build/` kept as the separate dataset-preprocessing folder; top README links it.
- [x] Old internal runbook preserved at `docs/README_runbook_legacy.md`.
- [ ] Add LICENSE file (pyproject + README badge declare Apache-2.0; file not yet present).
