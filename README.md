# OralSeg- open AI model for CBCT tooth segmentation

Based on SwinUNETR and Mamba frameworks.

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

## 2. Run training

Ensure the imported resources are in PYTHONPATH
(e.g., `./OralSeg/dataset_100`) then run the script:

```console
export PYTHONPATH=./OralSeg/dataset_100/
python OralSeg/main_dataset.py
```
