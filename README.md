# CUNet + CLSTM Based Liver Tumor Detection in CT Scans

A deep learning-based medical image segmentation system that automatically detects and segments liver tumors from CT scan images using a hybrid **CUNet + CLSTM** architecture.

---

## 🚀 Live Demo

**Streamlit Application:**  
https://cunetclstmliver-9twjsrhhwitbziq6ihrezu.streamlit.app/

> **Note:** The application is hosted on Streamlit Community Cloud. If the app is inactive, it may take a few seconds to wake up before loading.

---

## 📖 Overview

This project aims to automatically detect and segment liver tumors from CT scan images using deep learning. Early and accurate liver tumor detection plays a crucial role in supporting medical diagnosis and treatment planning.

The system uses the **LiTS (Liver Tumor Segmentation)** dataset, which contains abdominal CT scans and corresponding segmentation masks in **`.nii` (NIfTI)** format. The application directly accepts `.nii` files as input and performs preprocessing before passing them to a hybrid **CUNet + CLSTM** model for segmentation.

The segmented output highlights the liver and tumor regions, providing an efficient and user-friendly solution for medical image analysis through an interactive Streamlit web application.

---

## ✨ Features

- Automatic liver and tumor segmentation
- Supports CT scans in `.nii` format
- Hybrid CUNet + CLSTM architecture
- Interactive Streamlit web application
- Real-time prediction and visualization
- Medical image preprocessing pipeline

---

## 📂 Dataset

This project uses the **LiTS (Liver Tumor Segmentation)** dataset.

- Dataset Format: `.nii` (NIfTI)
- Image Type: Abdominal CT Scans
- Ground Truth: Liver and Tumor Segmentation Masks

> **Note:** The dataset is not included in this repository due to licensing restrictions and file size limitations.

---

## 🛠 Tech Stack

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

## 🏗 Project Structure

```text
CUNet-CLSTM-Liver-Tumor-Detection/
│
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

## 🔄 Workflow

1. Upload a CT scan in `.nii` format.
2. The image is preprocessed.
3. The trained CUNet + CLSTM model is loaded.
4. Liver and tumor segmentation is performed.
5. A prediction mask is generated.
6. The segmentation result is displayed through the Streamlit interface.

---

## ⚙️ Installation

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

## 🧠 Model Architecture

The proposed hybrid architecture combines:

- **CUNet** – Performs feature extraction and semantic segmentation using an encoder-decoder architecture.
- **CLSTM (Convolutional Long Short-Term Memory)** – Enhances contextual and spatial feature learning, improving tumor boundary segmentation.

Together, these components provide accurate liver and tumor segmentation from CT scan images.

---

## 📊 Evaluation Metrics

The model performance is evaluated using:

- Dice Coefficient
- Intersection over Union (IoU)

These metrics measure the overlap between the predicted segmentation mask and the ground truth.

---

## 🚀 Future Enhancements

- Improve segmentation accuracy with larger datasets.
- Add support for additional medical imaging formats.
- Deploy using Docker and Kubernetes.
- Integrate cloud-based prediction services.
- Add Grad-CAM visualization for model interpretability.

---

## 👨‍💻 Contributor

- Lalith Sai Muriki

---

## 📄 License

This project was developed for academic and educational purposes.
