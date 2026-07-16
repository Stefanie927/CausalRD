# CausalRD
This repository is an implementation of the paper 'Causal Representation Learning for Retinal Disease Diagnosis and Counterfactual Generation'

### 1. Dataset Preparation

Download the OIA-ODIR dataset from the official repository:

- Dataset: https://github.com/nkicsl/OIA-ODIR

After downloading the dataset, run the preprocessing script:

```bash
python image_cropping.py
```

The script performs retinal image preprocessing and generates the processed images. The preprocessed images will be saved to: data/OIA_ODIR/

### 2. Feature Extraction with RETFound-MAE

We use the pretrained retinal foundation model RETFound-MAE as the feature extractor.

The extracted feature representations for all samples are generated in advance and stored at: data/OIA_ODIR/ViT_features/

### 3. Training
```bash
python src/main_finetune.py
```

### 4. Multi-Disease Classification Evaluation
```bash
python src/test_classification.py
```

### 5. Image Generation
```bash
python src/generation.py
```

### 6. Counterfactual Image Generation
```bash
python src/gen_counterfactual.py
```
