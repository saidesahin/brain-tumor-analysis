# Explainable AI for Brain Tumor Analysis

An academic deep learning project focused on brain tumor classification using MRI sequences and Explainable Artificial Intelligence (XAI).

## Overview

This project explores deep learning-based brain tumor analysis using MRI scans. The main objective is to develop a classification model that distinguishes between tumor grades and provides visual explanations for its predictions.

The project evolved through two stages:

- **Tumor Type Prediction:** An initial three-class classification task to distinguish between Glioblastoma (GBM), Astrocytoma, and Oligodendroglioma.
- **Tumor Grade Prediction:** A subsequent binary classification task to distinguish between High-Grade Glioma (HGG) and Low-Grade Glioma (LGG), following challenges related to class imbalance in the three-class task.

## Dataset

The project uses the **UCSF-PDGM** dataset.

Four MRI sequences are used as model inputs:

- T1
- T1c (contrast-enhanced T1)
- T2
- FLAIR

Raw MRI scans, processed images, and clinical metadata are excluded from this repository to protect data privacy and comply with responsible data-handling practices.

Please refer to the original dataset documentation for access requirements and permitted use.

## Methodology

The project workflow consists of the following stages:

1. MRI data organization and preprocessing.
2. Preparation of model inputs using four MRI sequences.
3. Experimentation with multiple deep learning architectures.
4. Transition from three-class tumor type prediction to binary tumor grade prediction.
5. Evaluation and comparison of model performance.
6. Application of Explainable AI techniques to interpret model predictions.

## Model Development and Selection

Three deep learning architectures were evaluated to identify the best-performing model for the classification task:

- **EfficientNet-B3**
- **VGGNet16**
- **DenseNet-121**

The models were compared based on their classification performance. **EfficientNet-B3 achieved the best overall performance among the evaluated architectures and was selected for further development.**

### Classification Strategy

The initial approach focused on three-class tumor type prediction:

- Glioblastoma (GBM)
- Astrocytoma
- Oligodendroglioma

However, class imbalance made it difficult for the model to learn all three classes reliably, particularly the underrepresented classes.

To address this challenge, the project transitioned to binary tumor grade prediction:

- **High-Grade Glioma (HGG):** Glioblastoma
- **Low-Grade Glioma (LGG):** Astrocytoma and Oligodendroglioma

This reformulation grouped tumor types according to grade and enabled the development of a more suitable classification approach for the available data.

## Explainable Artificial Intelligence (XAI)

To investigate how the selected model makes predictions, the project incorporates explainability technique:

- **Grad-CAM:** Generates visual heatmaps highlighting image regions that contribute to the model's predictions.

This technique aims to make model behavior more interpretable and support further investigation of the relationship between MRI features and classification outcomes.

## Model Evaluation

Model performance is evaluated using standard classification metrics, including:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

The evaluation process supports model comparison and helps assess classification performance across the target classes.

## Technologies

- Python
- PyTorch
- EfficientNet-B3
- VGGNet16
- DenseNet-121
- Transfer Learning
- Grad-CAM
- Medical Image Processing

## Repository Structure

```text
brain-tumor-analysis/
├── tumor_grade_prediction/   # Source code
├── test_results.xlsx
├── .gitignore                # Excluded files and directories
└── README.md                 # Project documentation
```

Raw data, processed MRI scans, clinical metadata, model checkpoints, and potentially sensitive generated outputs are intentionally excluded from the public repository.

## Privacy and Responsible Use

This repository is intended to share source code and project documentation rather than patient data.

MRI data, clinical metadata, and other potentially sensitive materials are excluded. Any additional files should be reviewed before publication to ensure that they contain no personal, clinical, or restricted information.

Dataset use must comply with the original provider's terms and applicable requirements.

## Limitations and Disclaimer

This project is an academic research and development project. Its predictions and visual explanations are experimental and have not been established as a substitute for professional medical judgment, clinical diagnosis, or treatment decisions.

Further validation on independent datasets is necessary before considering clinical use.

## Author

Software Engineering student interested in Artificial Intelligence, Machine Learning, Deep Learning, and Medical Image Analysis.
