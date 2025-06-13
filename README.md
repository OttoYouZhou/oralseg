# OralSeg- open AI model for CBCT tooth segmentation

Based on SwinUNETR and Mamba frameworks.

## Installing

## 1. Create conda environment and install dependencies

Create Conda environment

```console
conda env create -n oralseg
conda activate oralseg
```

Install python dependencies that need to be built

```console
pip install -r requirements.txt
```

## 2. Run project

Ensure the imported resources are in PYTHONPATH
(e.g., `./OralSeg`) then run the script:

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
