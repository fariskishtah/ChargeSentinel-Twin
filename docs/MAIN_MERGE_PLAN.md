# Main merge plan — prepared, not executed

**Source:** `feature/faris-integration` at the final gate's reviewed and CI-passing HEAD. **Target:** `main` after `git fetch --all --prune`. The pre-gate target was `origin/main` at `f2c44cd95b29cefbde33b38181c8c8e8d6298621`. Re-check both SHAs immediately before any authorized merge. This document does not authorize merging.

The source history from `origin/main` through the reviewed pre-gate commit `338c977e061df646717d4156182d2af3c399b003` contains these exact commits, in order:

```text
6f268d3  shared integration skeleton
3a1c68e  API contracts and MLflow integration skeleton
07ec0cb  Streamlit dashboard and API client
e646a7b  Digital Twin simulator
58d5b32  API/UI/MLflow Docker stack
7cd029e  Streamlit Docker import fix
c7684c0  dashboard polish
c7b6d43  charging station simulator
6bd4624  Waseem demand integration
673e866  solar and session-anomaly integration
a85a50f  Fatima PR reconciliation and Waseem evidence restore
1fc2565  grid-load interval screen
1e8f996  Samsung documentation
0714530  Python 3.12 CI workflow
f5a1a44  validation and merge rehearsal
338c977  pip cache path fix
```

Include the final evidence-only commit(s) after `338c977` by freezing and recording `git rev-parse feature/faris-integration` at merge review. Confirm the exact full list with `git log --reverse --oneline origin/main..feature/faris-integration`; do not accept a branch that moved after review.

**Excluded branches:** `origin/person2-demand-forecasting` (superseded and dangerous), `feature/fatima-anomaly` PR #1 (closed without merge), `feature/waseem-demand`, `feature/mazen-solar`, and `feature/mariam-availability` are not separately merged. Their accepted evidence is selectively included in the Faris branch where applicable. Mariam's runtime source and a separately identified artifact rebuild are included; her source branch is unchanged.

**Before merge:** obtain team review of `docs/SAMSUNG_COMPLIANCE_MATRIX.md`, the proxy anomaly limitation, portable receipts, CI, and fresh-clone demo. Record `git rev-parse origin/main` and `git rev-parse feature/faris-integration`; create a local backup ref or an annotated pre-merge tag pointing to the reviewed main SHA only when the merge is authorized. Do not create a release tag during this gate.

**Merge rehearsal/merge:** the earlier local no-commit rehearsal had zero conflicts (`docs/RELEASE_REHEARSAL.md`), before Mariam integration. Once authorized, use a reviewed `--no-ff` merge so the merge commit is reversible; do not force-push. Run Python tests, Waseem and Mariam owner tests, Docker build/health/smoke, and verify demand, both solar horizons, session risk, grid-load, historical availability/recommendation 200, Digital Twin and simulator. Confirm CI on the resulting main commit before considering a release tag.

**Rollback after an authorized no-ff merge:** `git revert -m 1 <merge-commit-sha>` on `main`, then run the same tests and push the revert normally. Do not use `git reset --hard` plus force-push on shared main. The pre-merge backup ref/tag is an audit point, not a serving artifact.
