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

## create conda environment for building and developing

```console
conda env create -n toothseg
```

## build mamba

```console
cd mamba
python setup.py bdist_wheel --dist-dir=../dist
cd ../dist
pip install mamba_ssm-2.2.2-cp310-cp310-linux_x86_64.whl
cd ..
```

## build causal-conv1d 

```console
cd causal-conv1d
python setup.py bdist_wheel --dist-dir=../dist
cd ../dist
pip install causal_conv1d-1.4.0-cp310-cp310-linux_x86_64.whl
cd ..
```
