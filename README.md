# toothseg - open AI model for CBCT tooth segmentation

Based on SwinUNETR and Mamba frameworks.

## Dependencies

- conda
  - python 3.10
- nvcc

The following git repositories are included as submodules:
- https://github.com/state-spaces/mamba
- https://github.com/Dao-AILab/causal-conv1d
- https://github.com/Project-MONAI/research-contributions/

Download them using:
```
git submodule update --init --recursive
```

## create conda environment and install dependencies

```console
conda env create -n toothseg
```

If you need to install additional dependencies, install them
using conda (`conda install <package>`) or pip (`pip install <package>`),
and then update the `environment.yml` file by running:

```console
conda env export | grep -v ^prefix: > environment.yml 
```

## build and install mamba

```console
cd mamba
python setup.py bdist_wheel --dist-dir=../dist
cd ../dist
pip install mamba_ssm-2.2.2-cp310-cp310-linux_x86_64.whl
cd ..
```

## build and install causal-conv1d 

```console
cd causal-conv1d
python setup.py bdist_wheel --dist-dir=../dist
cd ../dist
pip install causal_conv1d-1.4.0-cp310-cp310-linux_x86_64.whl
cd ..
```

## run project

Ensure `research-contributions/SwinUNETR/BTCV` is added
as a source directory (is in PYTHONPATH), then run:

```console
cd src
python main.py
```