# OralSeg- open AI model for CBCT tooth segmentation

Based on SwinUNETR and Mamba frameworks.

## Dependencies

- conda
  - python 3.10
- nvcc

The following git repositories are included as submodules:
- https://github.com/state-spaces/mamba
- https://github.com/Dao-AILab/causal-conv1d
- https://github.com/Project-MONAI/research-contributions

Download them using:
```
git submodule update --init --recursive
```

## Installing

## 1. Create conda environment and install dependencies

```console
conda env create -n oralseg
```

### 1.1 Install additional dependencies

If you need to install additional dependencies, install them
using conda (`conda install <package>`) or pip (`pip install <package>`),
and then update the `environment.yml` file by running:

```console
pip install torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install packaging wheel
pip install causal_conv1d --no-build-isolation
pip install mamba_ssm --no-build-isolation
pip install monai[pynrrd]
```

```console
conda env export | grep -v ^prefix: > environment.yml
```

## 2. Run project

Ensure the imported resources are in PYTHONPATH
(e.g., `research-contributions/SwinUNETR/BTCV`) then run the script:

```console
export PYTHONPATH=./src/:./SwinUNETR/BTCV/
python src/main_dataset.py
```

## Troubleshooting

### RuntimeError: received 0 items of ancdata

Your OS open file limit is probably causing this error.
Check with `ulimit -a`, open files should be at least 4096.

Increase the limit to fix this error, see
https://stackoverflow.com/questions/71642653/how-to-resolve-the-error-runtimeerror-received-0-items-of-ancdata.
