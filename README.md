# OralSeg- open AI model for CBCT tooth segmentation

Based on SwinUNETR and Mamba frameworks.

## Dependencies

- conda
  - python 3.13
- nvcc

## Installing

## 1. Create conda environment and install dependencies

Create Conda environment

```console
conda env create -n oralseg
conda activate oralseg
```

Install python dependencies that need to be built

```console
pip install causal_conv1d==1.5.0.post8 mamba_ssm==2.2.4 --no-build-isolation
```

### 1.1 Install python dependencies and create environment.yml file (alternative)

If you need to install additional dependencies, install them
using conda (`conda install <package>`) or pip (`pip install <package>`),
and then update the `environment.yml` file by running:

```console
pip install torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install packaging wheel monai[pynrrd] nibabel
```

```console
conda env export | grep -v ^prefix: > environment.yml
```

## 2. Run project

Ensure the imported resources are in PYTHONPATH
(e.g., `./SwinUNETR/BTCV/`) then run the script:

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
