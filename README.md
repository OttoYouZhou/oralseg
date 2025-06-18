# OralSeg: Open AI model for CBCT tooth segmentation

**OralSeg** is a deep learning-based open-source framework for 3D instance segmentation of teeth, jaw bones, and mandibular canals in dental cone-beam computed tomography (CBCT) scans. The model combines the strengths of SwinUNETR and Mamba architectures, offering accurate and scalable segmentation for clinical and research use.

# Installation

## 1. Create Conda Environment

```console
conda env create -n oralseg python=3.10
conda activate oralseg
```

## 2. Install Dependencies
Install python dependencies that need to be built

```console
pip install -r requirements.txt
```

## 3. Data Preparation
### 3.1 Label Index
Each CBCT volume includes 35 segmentation labels, corresponding to individual teeth, jaw bones, and the mandibular canal. The label indices are as follows:

```bash
Maxilla (1)
Mandible (2)
Teeth (3–34):
  Upper: 11–18 (3–10), 21–28 (11–18)
  Lower: 31–38 (19–26), 41–48 (27–34)
Mandibular Canal (35)
```

### 3.2 Folder Structure
Place your data in the following directory layout:

```bash
OralSeg/
├── dataset/
│   ├── imagesTr/         # Input images in NIfTI format (*.nii.gz)
│   ├── labelsTr/         # Ground-truth labels (*.nii.gz)
│   └── dataset.json      # Training/validation configuration
```

### 3.3 Dataset JSON Format
The dataset.json file defines the training/validation split and label metadata. It should follow the MONAI standard format. Example content:

```bash
{
  "name": "OralSeg",
  "description": "CBCT tooth segmentation dataset",
  "labels": {
    "0": "background",
    "1": "maxilla",
    "2": "mandible",
    "3": "11", "4": "12", "5": "13", "6": "14", "7": "15", "8": "16", "9": "17", "10": "18",
    "11": "21", "12": "22", "13": "23", "14": "24", "15": "25", "16": "26", "17": "27", "18": "28",
    "19": "31", "20": "32", "21": "33", "22": "34", "23": "35", "24": "36", "25": "37", "26": "38",
    "27": "41", "28": "42", "29": "43", "30": "44", "31": "45", "32": "46", "33": "47", "34": "48",
    "35": "mandibular_canal"
  },
  "training": [
    {"image": "./imagesTr/case_001.nii.gz", "label": "./labelsTr/case_001.nii.gz"},
    {"image": "./imagesTr/case_002.nii.gz", "label": "./labelsTr/case_002.nii.gz"}
  ],
  "validation": [
    {"image": "./imagesTr/case_101.nii.gz", "label": "./labelsTr/case_101.nii.gz"}
  ]
}
```

## 2. Training

Ensure the imported resources are in PYTHONPATH
(e.g., `./OralSeg/dataset_100`) then run the script:

```console
export PYTHONPATH=./OralSeg/dataset/
python OralSeg/main_dataset.py
```
