# Fork research integration — 2026-10-25

AI-use disclosure: Codex prepared this handoff under Hao's direction.

## Current fork state

`research/rl-bot` consolidates `repro/ubuntu-cpu-smoke` at615e5f7f71380757d1c1f8ddc1db17d6d1079bb8, based on public `rl`9a9ecc9512cdc066f8c97a86b7ce5ca61ce8eacf. No divergent merge, upstream contribution, main change or release is part of this handoff. Fork main5912a1ace3099ac51069e8a5684f1090ae6538c8 is a different MPC line; preserve it and other upstream branches.

[Engineering evidence](RESULTS-2026-10-25.md): isolated Rust build and60tests passed,8192-step seed11 smoke completed two critic-only updates in6.431s; no actor-loss updates,100-game strength evaluation or browser behavior review. Actual source is schema20/1010 features plus mask/18actions. Upstream README's schema24/v16 performance claims are historical source context, not achieved fork results. Preserve upstream MIT and notices.

## Workflow and pending proposal

Future task branches start from and return to `research/rl-bot` after scoped verification/review. Main integration requires a separate explicit decision. Mac authors/commits; Ubuntu only synchronizes an approved branch/full HEAD and runs its frozen isolated environment. No new SSH sync, install, training or game is authorized by this handoff. Keep `.killfield-runtime` separate from Tank's runtime and retain all checkpoints/metrics/backup manifests. Old qualified source identities remain necessary for reproduction.

Recommended next decision: fresh seed11 in the original schema20 environment, proposed256envs×128×64updates=2097152steps (20critic warmup,44actor updates);50/50Laika/MPC with absent ancestor explicitly omitted, original200m schedule preserved. Training45min/step cap first; evaluation/loading/cleanup15min; viewer15min/max3 observation games; separate setup20min; cloud$0;3GiB scoped runtime and10GiB free-disk reserve. These caps are pending approval, not measured throughput or promised learning.

This is an endpoint-behavior probe, not a learning-gain estimate: no step0 panel is budgeted. Proposed checkpoint cadence is each complete update. Bind the normal64-update endpoint model/manifest hashes explicitly; if training hits its cap early, preserve/report incomplete and do not substitute older live.pt for the100-game panel. Additional diagnostics need approval.

Exactly100Laika evaluation games, deterministic argmax, proposed seed9000 and oneenv to avoid batch overshoot; local loopback viewer over SSH. Report model identity, exact observation/action/reward and all outcomes regardless of quality. Follow AGENTS' human behavior review; no performance gate that blocks showing a completed model. Viewer WASM target/setup is not yet installed or approved.

Before execute, resolve bounded continuation: upstream resume retains optimizer/schedule, not RNG/engine, skips warmup and overwrites same-output metrics. Fresh attempt paths; same experiment and remaining budgets; stop for review if interrupted before warmup completes; otherwise explicit latest verified complete checkpoint only. No smoke initialization, automatic retry or budget reset. No trainer framework rewrite.

[Cross-repository assessment](https://github.com/JarrettChen217/tank-trouble-reconstruction/blob/research/rl-bot/docs/evidence/rl-research-integration-2026-10-25.md) and [Asana coordination](https://app.asana.com/1/1219142319278837/project/1219141547723528/task/1219870066225957) track protocol review → Hao approval → bounded execution → actual-versus-expected review → optional Tank adaptation → separately authorized main promotion. No GPU/holdout/cloud/web migration now. No current SSH state is asserted; recheck before future execution.
