# Project rules for every agent

## Start here

**Read [`docs/HANDOFF.md`](docs/HANDOFF.md) before anything else.** It is the
authoritative description of what this branch currently trains: the duel
curriculum, schema 20, `Discrete(18)`, the reward, the opponent pool, where the
checkpoints are, and what has already been verified so you do not re-verify it.

`README.md`, `docs/DESIGN.md` and `docs/TRAINING.md` describe the same current
design. `docs/LESSONS.md` is history: every curriculum it names is retired, and
it is kept for the conclusions those experiments falsified, not as a spec.

## Mandatory training handoff

For the current PPO route:

1. Do not add an evaluation gate or block deployment on a metric threshold.
2. Every completed training run must be exposed in the web viewer for direct behavior review.
3. Evaluate exactly 100 games against the fixed Laika opponent for the reported win rate. Do not
   silently replace this with a larger sample or a gate.
4. Every training handoff must explicitly report:
   - the model/checkpoint name;
   - the exact reward;
   - the exact observation;
   - the exact action space;
   - the win rate over exactly 100 games against Laika;
   - the browser command/URL used to review behavior.

These are reporting and review requirements, not promotion criteria. The user decides behavior
quality by watching the model.

## Fork research workflow (Hao)

- Read `repro/RESEARCH_HANDOFF.md` for this fork's current state. Upstream documents are historical context where they conflict with the fork's measured evidence.
- Use `research/rl-bot` as the research integration branch. New task branches start there and integrate there after scoped validation/review. Main promotion and publication require separate explicit human authorization. Do not automatically merge the upstream MPC main and RL line.
- Mac edits/commits code, tests, configuration and locks. Before approved target work, Ubuntu must match the same branch/full HEAD and source fingerprints with clean tracked files; use frozen isolation under sibling `.killfield-runtime`. Do not change system Python/drivers/shell or install coding agents. Preserve unexpected changes and report; no automatic reset/stash/merge/cleanup.
- Freeze expected outcome, workload/time/disk budget, comparison, continuation and stop rules before Hao authorizes execution. Report actual versus expected results, preserve interruptions and all artifacts, then stop for the next decision. Source integration does not approve a training run or establish learned strength.
