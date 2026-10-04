# Data generation

Render, reconstruct and preprocess your own animated assets into an AeroSplat-4D-style training
root. The source 3D models of the paper are not redistributed.

## 1. Render (Isaac Sim 4.5 or 5.1)

Add one entry per `.usd`/`.usdc` asset to `isaacsim/configs/asset_config.yaml` (the examples show
the format and the 1 m normalisation), list it under `assets:` in `isaacsim/configs/config_batch.yaml`,
and set `AEROSPLAT_ASSET_ROOT`, `AEROSPLAT_RENDER_ROOT` and `ISAAC_SIM_PATH`:

```bash
pipelines/isaacsim/run_batch.sh --dry-run   # list the sweep
pipelines/isaacsim/run_batch.sh             # re-run to resume; --no-resume renders everything again
```

Output: `$AEROSPLAT_RENDER_ROOT/aerosplat4d_renders/<class>/<sequence>/`. Later stages parse camera
parameters from the sequence directory name, so do not rename it.

## 2. Reconstruct with LGM

```bash
git clone https://github.com/3DTopia/LGM && cd LGM
git checkout fe8d12cff8c827df7bb77a3c8e8b37408cb6fe4c
git apply /path/to/MambaSplat-4D/pipelines/lgm/lgm.patch
pip install --no-build-isolation git+https://github.com/ashawkey/diff-gaussian-rasterization
pip install -r requirements.txt kiui==0.3.3
mkdir pretrained && wget -P pretrained https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors
cd .. && LGM_ROOT=$PWD/LGM python pipelines/lgm/batch_reconstruct.py \
    --input $AEROSPLAT_RENDER_ROOT/aerosplat4d_renders \
    --output $AEROSPLAT_RENDER_ROOT/aerosplat4d_lgm_ply \
    --checkpoint LGM/pretrained/model_fp16_fixrot.safetensors
```

`kiui` is pinned because 0.3.5 fails to import (`NameError: name 'Union' is not defined`).

## 3. Preprocess

```bash
python -m mambasplat4d.preprocessing.preprocess_aerosplat4d \
    --input  $AEROSPLAT_RENDER_ROOT/aerosplat4d_lgm_ply/data \
    --output $AEROSPLAT_DATA_ROOT/preprocessed_full \
    --split-yaml my_split.yaml --n-versions 1 --seed 42
python -m mambasplat4d.preprocessing.subsets.build_lgm_4class \
    --src $AEROSPLAT_DATA_ROOT/preprocessed_full \
    --dst $AEROSPLAT_DATA_ROOT/preprocessed_50-25-25_4class \
    --split-yaml my_split.yaml
```

The split YAML lists every asset on disk under its class, in `train`, `val` and `test` lists
(format: `assets/splits/open_split.yaml`). `build_lgm_4class` expects the full radius and elevation
sweep; with a different sweep, use the `preprocess_aerosplat4d` output as the training root directly.
