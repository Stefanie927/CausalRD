# CausalRD
This repository is an implementation of the paper 'Causal Representation Learning for Retinal Disease Diagnosis and Counterfactual Generation'

### 1. Dataset Preparation

Download the OIA-ODIR dataset from the official repository:

- Dataset: https://github.com/nkicsl/OIA-ODIR

After downloading the dataset, run the preprocessing script:

```bash
python image_cropping.py
```

The script performs retinal image preprocessing and generates the processed images. The preprocessed images will be saved to: .data/OIA_ODIR/

### 2. Feature Extraction with RETFound-MAE

We use the pretrained retinal foundation model RETFound-MAE as the feature extractor.

The extracted feature representations for all samples are generated in advance and stored at: .data/OIA_ODIR/ViT_features/
