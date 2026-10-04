#!/bin/bash

# Render AeroSplat-4D acquisition sweep.
#   ./run_batch.sh --dry-run                # preview jobs
#   ./run_batch.sh                          # render jobs lacking .render_complete
#   ./run_batch.sh --asset-type bird        # birds only
#   ./run_batch.sh --assets "crow.usdc"     # selected assets
#   ./run_batch.sh --no-resume              # re-render finished jobs
#   ./run_batch.sh --list-assets            # list assets in asset_config.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-$HOME/isaacsim}"

if [ ! -f "$ISAAC_SIM_PATH/python.sh" ]; then
    echo "ERROR: python.sh not found in $ISAAC_SIM_PATH"
    echo "Set ISAAC_SIM_PATH to your Isaac Sim installation directory"
    echo "Example: export ISAAC_SIM_PATH=~/isaacsim"
    exit 1
fi

echo "Using Isaac Sim at: $ISAAC_SIM_PATH"
export ISAAC_SIM_PATH

# host python; launches runners/smart_batch_runner.py via Isaac Sim python.sh
python3 "$SCRIPT_DIR/runners/batch_render.py" --isaac-sim-path "$ISAAC_SIM_PATH" "$@"
