#!/bin/bash
# Clone DP-AggZO at the pinned commit and apply the patches in baselines/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$ROOT/baselines/dp-aggzo"
PINNED_COMMIT=99f64026cf744be804da3dedfe025a6e199df82f

[[ -d "$TARGET/.git" ]] || git clone https://github.com/erguteb/dp-aggzo.git "$TARGET"
git -C "$TARGET" reset --hard "$PINNED_COMMIT"
for name in fixes unique_tag bos_probe loss_chunk transformers445 schedule_eval; do
  git -C "$TARGET" apply "$ROOT/baselines/dp_aggzo_$name.patch"
done
echo "[setup] DP-AggZO at $PINNED_COMMIT with the six patches applied"
