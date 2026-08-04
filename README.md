# CUNet + CLSTM Based Liver Tumor Detection in CT Scans

## Overview

This project presents a deep learning-based approach for automatic liver tumor detection and segmentation from CT scan images. The system combines **CUNet** and **CLSTM (Convolutional Long Short-Term Memory)** to improve segmentation accuracy by capturing both spatial and contextual features.

The application processes CT scans in **NIfTI (.nii)** format and generates segmentation masks highlighting liver and tumor regions. An interactive **Streamlit** web application is provided for easy prediction and visualization.

---

## Features

- Automatic liver and tumor segmentation
- Supports CT scans in `.nii` format
- Hybrid CUNet + CLSTM architecture
- Interactive Streamlit web interface
- Real-time prediction and visualization
- Medical image preprocessing pipeline

---

## Dataset

This project uses the **LiTS (Liver Tumor Segmentation)** dataset.

- Dataset Format: `.nii` (NIfTI)
- Image Type: Abdominal CT Scans
- Ground Truth: Liver and Tumor Segmentation Masks

> **Note:** The dataset is not included in this repository due to its size and licensing restrictions.

---

## Technology Stack

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- NiBabel
- Matplotlib
- Streamlit
- Git
- GitHub
- Git LFS

---

## Project Structure

```
├── dataset/
├── models/
├── preprocessing/
├── training/
├── prediction/
├── app.py
├── requirements.txt
├── README.md
└── trained_model.h5
```

---

## Workflow

1. Upload a CT scan in `.nii` format.
2. The image is preprocessed.
3. The trained CUNet + CLSTM model is loaded.
4. Liver and tumor segmentation is performed.
5. A prediction mask is generated.
6. Results are displayed through the Streamlit interface.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/CUNet-CLSTM-Liver-Tumor-Detection.git
cd CUNet-CLSTM-Liver-Tumor-Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

## Model

The proposed hybrid architecture consists of:

- **CUNet** for feature extraction and segmentation.
- **CLSTM** for enhancing contextual and spatial feature learning.

The trained model generates accurate liver and tumor segmentation masks from CT scan images.

---

## Evaluation Metrics

The model performance is evaluated using:

- Dice Coefficient
- Intersection over Union (IoU)

---

## Results

The application successfully segments liver tumors from CT scans and provides clear prediction masks through an interactive web interface.

---

## Future Enhancements

- Improve segmentation accuracy with larger datasets.
- Support additional medical imaging formats.
- Deploy using Docker and Kubernetes.
- Integrate cloud-based prediction services.
- Add Grad-CAM visualization for model interpretability.

---

## Contributors

- Lalith Sai Muriki
- Team Members

---

## License

This project is developed for academic and educational purposes.
