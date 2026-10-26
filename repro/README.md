# Isolated Ubuntu engineering reproduction

AI-use disclosure: Codex prepared this isolated setup under Hao's explicit
authorization. The upstream engine, reward, observations, policy and optimizer
are preserved. This is an engineering smoke, not mature Hybrid reproduction.

- Upstream: Cichlider/killfield, `rl`, full
  `9a9ecc9512cdc066f8c97a86b7ce5ca61ce8eacf`; retain upstream MIT notices.
- Separate fork/checkout: JarrettChen217/killfield; branch
  `repro/ubuntu-cpu-smoke`. Tank Trouble's checkout/runtime is out of scope.
- Only upstream-source adaptation: select the native Linux `.so` library name;
  Darwin keeps its original `.dylib` path. No game/training behavior change.
- Exact upstream NumPy 2.5.2 requires Python >=3.12. Use uv-managed CPython 3.12
  separately from Tank Trouble's 3.11. Torch 2.13.0 is the CPU-only distribution.
- `repro/uv.lock` binds the separate Linux runtime. Do not install on the host
  interpreter or install CUDA, a coding agent, ROS, Node/browser or wasm tools.
- Ubuntu runtime root: sibling `.killfield-runtime/`; set UV_CACHE_DIR,
  UV_PYTHON_INSTALL_DIR, UV_PROJECT_ENVIRONMENT, TMPDIR, XDG_CACHE_HOME,
  CARGO_HOME, RUSTUP_HOME and CARGO_TARGET_DIR explicitly under it. Minimal Rust
  1.88.0; no shell/profile changes or default `~/.cargo`/`~/.rustup` writes.
- Mac authors/commits setup; Ubuntu fetches this fork branch. Before execution,
  require matching full HEAD, branch and tracked-file SHA inventory.
- Preflight disk reserve 10 GiB; stop setup if scoped storage exceeds 3 GiB.
  Preserve files/outputs on failure; no cleanup/reset/stash of existing work.

## Bounded checks

1. Frozen isolated runtime sync, locked native library build and upstream
   `cargo test --locked --lib`. No full performance benchmark or GUI service.
2. Author's documented engineering smoke: seed11, 32 envs, 8192 steps/two
   updates, 128 rollout frames per env, 4 epochs/8 minibatches, at most300s.
   Use `--mix 1 0 0 --threads 4` because no original frozen ancestor is public.
   This explicit engineering-only opponent pool is not the production40/40/20.
3. Keep the author's critic warmup20 unchanged. Both smoke updates are warmup;
   do not claim that a trained actor or original strength was reproduced.
4. Save final/live/resume bundles and metrics under a new ignored
   outputs/repro-smoke path. Check finite observations/losses, schema20/1010/18,
   exact two updates and checkpoint readable. Do not restart a failed smoke.
5. No 100-game strength report, viewer service, deployment, long training, GPU
   or cloud. Upstream's full training handoff requirements therefore remain
   unfulfilled; do not label this smoke a completed behavioral training run.
6. Report actual disk use, checks, deviations and protected persistent-file
   fingerprints. Preserve upstream docs rather than overwriting their history.


## Completed week1 reference validation

The bounded delegated study completed October26,2026; see [WEEK1_RESULTS](WEEK1_RESULTS.md). Actual100-game schema20 rates: seed11step0/u64/u2563%/11%/26%, seed22/33u648%/9%. Generic tools/config/tests/analysis live in Tank; this fork only holds minimal reference trainer/evaluator hooks. All declared500cases/8viewer observations complete, jobs closed; no automatic next run or main/publication acceptance. Earlier smoke remains historical engineering only.
