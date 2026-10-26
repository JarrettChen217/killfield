# Week1 reference validation — complete October26,2026

AI-use disclosure: Codex coordinated this bounded reference validation under Hao's delegation.

Consolidated original tools/config/tests/analysis and evidence belong in Tank:
[results](https://github.com/JarrettChen217/tank-trouble-reconstruction/blob/research/rl-bot/docs/evidence/rl-week1-results-2026-11-01.md),
[status](https://github.com/JarrettChen217/tank-trouble-reconstruction/blob/research/rl-bot/docs/RL_STATUS.md),
[validation](https://github.com/JarrettChen217/tank-trouble-reconstruction/blob/research/rl-bot/docs/evidence/rl-week1-validation-2026-10-26.json).

Execution pinned Tankfdb36e6fafb0c88ff59f6b586a7d15123ab5a1dc / forkf870046187bdc9707b76f9753c7d0f361612cd3a, same clean taskbranch on Mac/Ubuntu. This fork retained only minimal upstream trainer/evaluator hooks; original engine/network/PPO/reward/savers/Adam/schedule/viewer reused, MIT/provenance intact.

| Model | Win/loss/double/draw over exactly100Laika argmax cases |
| --- | --- |
|11step0|3/89/7/1 (3%)|
|11u64|11/73/16/0 (11%)|
|22u64|8/71/21/0 (8%)|
|33u64|9/74/16/1 (9%)|
|11u256|26/34/40/0 (26%)|

Cases9000+i/env1,500started=500complete. Three fresh64-update runs and the eligible predeclared11+192 continuation used12,582,912submitted decisions/4222.955s,CPU,256envs,schema20,50/50Laika/MPC,noancestor,original200m schedule. This is not mature schema24/v17b reproduction. Latestu256 modelSHA1d6f055290eac1e0afe2a7f17a343cfdd8429ef1f8d5cb8bdda69b93b2d7529f.

Every trained endpoint was shown in a bounded original loopback viewer:8games total/278.815s, observerCodex, no human acceptance. Commands/URLs/OAR/checkpoint identities are in the Tank report and ignored receipts. StaticMPC button highlight was a recorded display caveat; actualPPO binding verified and Tank staginglabel fixed afterward without new games.

New3-update engineering interruption/load/continue qualified warmup/Adam/schedule in41.139s/98,304steps; its weights never initialized formal runs.392completebundles verified;1055selectedfiles/43,425,374B SHA-backedupMac, allrawretained. No active research process/server, auto-restart service or scheduled loop. Main, publication, Tank adaptation/training and human/course decisions remain separate. Direct weight port deferred; reference26% does not establish Tank compatibility.
