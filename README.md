# Brain Tumor MRI Segmentation

**Deep Learning | Medical Image Segmentation | TensorFlow | U-Net | ResNet50 | Streamlit**

An end-to-end deep learning project exploring brain tumor segmentation in MRI images, from data preparation and model benchmarking to performance analysis and interactive deployment.

The project compares several U-Net-based architectures and training strategies, with particular attention to segmentation performance across tumor categories and difficult glioma cases.

> **Research and educational use only.** This project is not a medical device, has not been clinically validated, and must not be used for diagnosis, treatment planning, or medical decision-making.

## 🚀 Live Demo

**[Launch the Interactive Streamlit Application](https://brain-tumor-mri-segmentation-fsy.streamlit.app/)**

The application provides three interactive sections:

- **Model Performance:** Explore validation results and model comparisons.
- **Explore MRI Samples:** Compare original MRI images, ground-truth masks, predicted masks, and segmentation overlays.
- **Upload Research MRI:** Upload a non-identifiable research MRI image and visualize the model's predicted segmentation.

The trained model is hosted separately on Hugging Face.

**[View Model on Hugging Face](https://huggingface.co/read855/brain-mri-segmentation-resnet50-unet)**

## 🎯 Project Objectives

This project investigates three main questions:

1. How effectively can U-Net-based architectures segment brain tumor regions in MRI images?
2. How do data augmentation, loss-function optimization, and a pretrained encoder affect segmentation performance?
3. What can category-level evaluation and failure analysis reveal beyond a single aggregate performance metric?

The objective is not only to improve segmentation scores, but also to build a reproducible experimental workflow and communicate model limitations transparently.

## 📊 Dataset

**Dataset:** BRISC2025 - Brain Tumor MRI Dataset

**Source:** [Kaggle - BRISC2025](https://www.kaggle.com/datasets/briscdataset/brisc2025)

**License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

The project uses MRI images and corresponding segmentation masks across three tumor categories:

- Glioma
- Meningioma
- Pituitary tumor

### Dataset Split

| Split | Number of Images |
|---|---:|
| Training | 3,345 |
| Validation | 588 |
| Test | 853 |

The validation set includes:

| Tumor Category | Validation Images |
|---|---:|
| Glioma | 160 |
| Meningioma | 211 |
| Pituitary | 217 |
| **Total** | **588** |

Images and masks are processed at a resolution of **256 × 256 pixels**.

**Dataset attribution:** BRISC2025, distributed through Kaggle under CC BY 4.0. Images used in this project have been resized and processed for segmentation experiments. The interactive application includes a small selection of example images and reference masks derived from the dataset.

## 🧠 Methodology

### 1. Data Preparation

MRI images and segmentation masks are prepared for supervised semantic segmentation.

The workflow includes:

- Image resizing and preprocessing
- Binary segmentation mask preparation
- Training and validation data organization
- Identical-image checks using SHA-256 hashes

### 2. Baseline U-Net

A standard U-Net architecture establishes the initial segmentation benchmark.

### 3. Data Augmentation

An augmented U-Net experiment investigates whether greater training-image variability improves segmentation performance.

### 4. Loss-Function Optimization

A Tversky-based optimization strategy explores the handling of segmentation errors and difficult cases.

### 5. Transfer Learning

A ResNet50 U-Net combines a pretrained ResNet50 encoder with a segmentation decoder.

The pretrained encoder remains frozen during training.

### 6. Error Analysis

Performance is evaluated using the Dice coefficient, including category-specific results and an analysis of severe glioma segmentation failures.


## 📈 Experimental Results

Four U-Net-based models were evaluated using the Dice coefficient, Intersection over Union (IoU), precision, and recall.

Model development and selection were based on the validation set (588 images). The selected models were subsequently evaluated on the test set (853 images) using the same preprocessing pipeline and a binary segmentation threshold of 0.5.

### 1. Validation Performance — Model Selection

| Model | Mean Dice | Glioma Dice | Severe Glioma Failures |
|---|---:|---:|---:|
| Baseline U-Net | 0.8353 | 0.6830 | 22 / 160 |
| Augmented U-Net | 0.8417 | 0.7004 | 17 / 160 |
| Tversky U-Net | 0.8510 | 0.7352 | 13 / 160 |
| **ResNet50 U-Net** | **0.8570** | **0.7595** | **10 / 160** |

*Validation set: 588 MRI images, including 160 glioma images.*

ResNet50 U-Net achieved the highest mean validation Dice and the highest glioma Dice. It was therefore selected as the final model for the interactive application.

### 2. Test Performance — Overall Results

The four saved model checkpoints were evaluated on the same 853 test images without additional training or model modification.

| Model | Mean Dice | Mean IoU | Mean Precision | Mean Recall |
|---|---:|---:|---:|---:|
| Baseline U-Net | 0.8209 | 0.7412 | **0.8580** | 0.8452 |
| Augmented U-Net | 0.8124 | 0.7309 | 0.8374 | 0.8439 |
| Tversky U-Net | 0.8269 | 0.7422 | 0.8052 | 0.8982 |
| **ResNet50 U-Net** | **0.8373** | **0.7556** | 0.8273 | **0.9042** |

ResNet50 U-Net maintained the highest overall Dice and IoU scores on the test set.

Its mean Dice decreased from **0.8570 on validation to 0.8373 on test**, a difference of 0.0197.

### 3. Test Performance — Results by Tumor Category

| Model | Glioma Dice | Meningioma Dice | Pituitary Dice |
|---|---:|---:|---:|
| Baseline U-Net | 0.6698 | **0.9256** | **0.8432** |
| Augmented U-Net | 0.6693 | 0.9154 | 0.8295 |
| Tversky U-Net | 0.7117 | 0.9191 | 0.8312 |
| **ResNet50 U-Net** | **0.7364** | 0.9233 | 0.8356 |

The results reveal differences in segmentation performance across tumor categories:

- **Glioma:** ResNet50 U-Net achieved the highest Dice score (0.7364).
- **Meningioma:** Baseline U-Net achieved the highest Dice score (0.9256).
- **Pituitary:** Baseline U-Net achieved the highest Dice score (0.8432).

These results show that the best overall model does not necessarily perform best for every tumor category.

### 4. Severe Glioma Segmentation Failures

Glioma segmentation remained the most challenging category in this evaluation.

A severe lack of generalization is defined as a prediction with a Dice score below 0.20.

| Model | Severe Failures | Total Glioma Images |
|---|---:|---:|
| Baseline U-Net | 30 | 254 |
| Augmented U-Net | 31 | 254 |
| **Tversky U-Net** | **17** | 254 |
| ResNet50 U-Net | 20 | 254 |

Although ResNet50 U-Net achieved the highest average glioma Dice, Tversky U-Net produced fewer severe glioma failures on the test set.

### 5. Key Findings

- **ResNet50 U-Net achieved the best overall test Dice (0.8373)** and IoU (0.7556).
- The model also achieved the highest test Dice for glioma segmentation (0.7364).
- Baseline U-Net performed slightly better for meningioma and pituitary segmentation.
- Tversky U-Net achieved the lowest number of severe glioma failures (17/254).
- ResNet50 U-Net achieved the highest mean recall on the test set (0.9042).
- All four models obtained lower mean Dice scores on test than on validation.
  

## 🏗️ Final Model

**Architecture:** ResNet50 U-Net

| Property | Configuration |
|---|---|
| Task | Binary tumor segmentation |
| Input | 256 × 256 × 1 |
| Output | 256 × 256 × 1 |
| Encoder | Pretrained ResNet50 |
| Encoder training | Frozen |
| Loss strategy | Binary cross-entropy + Tversky |
| Training | 25 epochs |
| Parameters | Approximately 30.55 million |
| Framework | TensorFlow / Keras |
| Model format | `.keras` |

The model checkpoint is hosted on Hugging Face rather than committed to GitHub.

**Model repository:** [model repository/brain-mri-segmentation-resnet50-unet](https://huggingface.co/read855/brain-mri-segmentation-resnet50-unet)

## 🖥️ Interactive Application

The deployed Streamlit application makes the segmentation workflow accessible without requiring users to run the notebooks.

### Explore MRI Samples

Explore a curated selection of validation MRI images with four complementary views:

- Original MRI
- Ground-truth segmentation mask
- Predicted segmentation mask
- Prediction overlay

Individual Dice scores help illustrate variation between cases.

### Upload Research MRI

Upload a PNG or JPEG research image to generate a predicted segmentation mask and overlay.

**Important:** Uploaded images should be non-identifiable research images only. Predictions are experimental and must not be used for medical interpretation.

**[Try the Live Application](https://brain-tumor-mri-segmentation-fsy.streamlit.app/)**

## 📁 Repository Structure

```text
brain-tumor-mri-segmentation/
├── README.md
├── .gitignore
├── notebooks/
│   └── ...                       # Data preparation and experiments
├── artifacts/
│   └── ...                       # Experimental outputs
└── streamlit_app/
    ├── app.py                    # Interactive application
    ├── requirements.txt          # Application dependencies
    ├── .gitignore
    ├── samples/
    │   ├── samples_manifest.csv
    │   └── ...                   # Curated MRI examples
    └── ground_truth/
        └── ...                   # Corresponding reference masks
```

Model weights are not stored in this repository. The Streamlit application retrieves the checkpoint from Hugging Face when required.

## 🛠️ Technology Stack

- **Python** — Data processing and experimentation
- **TensorFlow / Keras** — Deep learning and model training
- **U-Net / ResNet50** — Segmentation architectures
- **NumPy / Pandas** — Data manipulation and analysis
- **Streamlit** — Interactive web application
- **Hugging Face Hub** — Model hosting
- **Google Colab** — Experimental environment
- **GitHub** — Source-code version control

## ⚠️ Limitations

### Segmentation of the glioma images

The final model still exhibits lack of generalization for the segmentation, particularly in challenging glioma images.


## 🔮 Potential Future Improvements

- Investigate alternative segmentation architectures and encoder configurations.
- Explore additional loss functions and strategies for difficult tumor cases.
- Assess robustness across different MRI acquisition settings.
- Extend error analysis to better characterize model uncertainty and failure patterns.
- Evaluate the model on an independent, patient-level-separated dataset.

## 📚 Dataset Attribution and License

**Dataset:** [BRISC2025 on Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025)

**Dataset license:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The dataset's original images and annotations belong to their respective creators and contributors.

This project uses and transforms dataset material for research and educational demonstration, including preprocessing, resizing, and the generation of example visualizations.

The dataset license applies to the source dataset.

---

**Live Application:** https://brain-tumor-mri-segmentation-fsy.streamlit.app/

**Model Checkpoint:** https://huggingface.co/read855/brain-mri-segmentation-resnet50-unet

**Repository:** https://github.com/nouraastafofana-droid/brain-tumor-mri-segmentation

*An experimental deep learning project focused on segmentation performance, transparent evaluation, and accessible model demonstration.*
