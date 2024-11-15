# toothseg - open AI model for CBCT tooth segmentation

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
conda env create -n toothseg
```

### 1.1 Install additional dependencies

If you need to install additional dependencies, install them
using conda (`conda install <package>`) or pip (`pip install <package>`),
and then update the `environment.yml` file by running:

```console
conda env export | grep -v ^prefix: > environment.yml
```

### 1.2 Build and install mamba

```console
cd mamba
git checkout v1.0.1
python setup.py bdist_wheel --dist-dir=../dist
cd ../dist
pip install mamba_ssm-2.2.2-cp310-cp310-linux_x86_64.whl
cd ..
```

(or just run `pip install .` in `mamba` if you don't need the wheel)

### 1.3 Build and install causal-conv1d 

```console
cd causal-conv1d
git checkout v1.0.0
python setup.py bdist_wheel --dist-dir=../dist
cd ../dist
pip install causal_conv1d-1.4.0-cp310-cp310-linux_x86_64.whl
cd ..
```

(or just run `pip install .` in `causal-conv1d` if you don't need the wheel)

## 2. Run project

Ensure the imported resources are in PYTHONPATH
(e.g., `research-contributions/SwinUNETR/BTCV`) then run the script:

```console
export PYTHONPATH=src:research-contributions/SwinUNETR/BTCV
python main.py
```