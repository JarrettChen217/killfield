#!/usr/bin/env bash
set -euo pipefail
test "$(uname -s)" = Linux
test "$(uname -m)" = x86_64
test -f repro/uv.lock
test -z "$(git status --porcelain)"
source repro/runtime.sh
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$KILLFIELD_RUNTIME/downloads"
test "$(df -PB1 "$KILLFIELD_RUNTIME" | awk 'NR==2 {print $4}')" -gt 10737418240
test "$(du -sB1 "$KILLFIELD_RUNTIME" | cut -f1)" -lt 3221225472
uv_bin=/home/haochen/.local/bin/uv
test -x "$uv_bin"
if ! test -x "$CARGO_HOME/bin/rustup"; then
    installer="$KILLFIELD_RUNTIME/downloads/rustup-init-1.28.2"
    url=https://static.rust-lang.org/rustup/archive/1.28.2/x86_64-unknown-linux-gnu/rustup-init
    curl --fail --location --silent --show-error "$url" -o "$installer"
    curl --fail --location --silent --show-error "$url.sha256" -o "$installer.sha256"
    expected=$(awk '{print $1}' "$installer.sha256")
    actual=$(sha256sum "$installer" | cut -d' ' -f1)
    test "$actual" = "$expected"
    chmod u+x "$installer"
    "$installer" -y --no-modify-path --profile minimal --default-toolchain 1.88.0
fi
"$CARGO_HOME/bin/rustc" --version
"$CARGO_HOME/bin/cargo" --version
"$uv_bin" sync --project repro --frozen --python 3.12.15 --no-dev
test "$(du -sB1 "$KILLFIELD_RUNTIME" | cut -f1)" -lt 3221225472
if ! test -e engine/target; then
    ln -s "$CARGO_TARGET_DIR" engine/target
fi
test "$(readlink -f engine/target)" = "$CARGO_TARGET_DIR"
"$CARGO_HOME/bin/cargo" build --locked --release --lib --manifest-path engine/Cargo.toml
"$CARGO_HOME/bin/cargo" test --locked --lib --manifest-path engine/Cargo.toml
"$UV_PROJECT_ENVIRONMENT/bin/python" -B -c 'import sys,numpy,torch; print(sys.version); print("numpy",numpy.__version__,"torch",torch.__version__,"cuda",torch.version.cuda); assert torch.version.cuda is None'
test "$(du -sB1 "$KILLFIELD_RUNTIME" | cut -f1)" -lt 3221225472
