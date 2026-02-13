
import streamlit as st
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from models.cunet_clstm import CUNet_CLSTM
import tempfile
import os

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="LiverAI - Tumor Detection",
    layout="wide"
)

# ----------------------------
# Glassmorphism CSS
# ----------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Glass Card */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
    margin-bottom: 25px;
    text-align: center;
}

/* Hide default header/footer */
header {visibility: hidden;}
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Device Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = CUNet_CLSTM().to(device)
    model.load_state_dict(torch.load("cunet_clstm_model.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# ----------------------------
# Header Section
# ----------------------------
st.markdown("""
<div class="glass-card">
<h1>🏥 LiverAI - CUNet + CLSTM</h1>
<h4>Advanced Deep Learning Liver Tumor Detection System</h4>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload NIfTI (.nii) CT Scan", type=["nii"])

if uploaded_file is not None:

    with st.spinner("🧠 Running AI Analysis..."):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        volume = nib.load(tmp_path).get_fdata()

        volume = np.clip(volume, -200, 250)
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

        depth = volume.shape[2]
        seq_len = 16

        center = depth // 2
        start = max(0, center - seq_len // 2)
        end = min(depth, start + seq_len)

        volume_seq = volume[:, :, start:end]

        if volume_seq.shape[2] < seq_len:
            pad = seq_len - volume_seq.shape[2]
            volume_seq = np.pad(volume_seq, ((0,0),(0,0),(0,pad)))

        volume_seq = torch.tensor(volume_seq, dtype=torch.float32)\
                        .permute(2,0,1).unsqueeze(1).unsqueeze(0).to(device)

        with torch.no_grad():
            seg_out, cls_out = model(volume_seq)

        tumor_prob = cls_out.item()

        seg_out = seg_out.squeeze().cpu().numpy()
        seg_mask = (seg_out > 0.3).astype(np.float32)

        os.remove(tmp_path)

    # ----------------------------
    # Metrics Section
    # ----------------------------
    tumor_percentage = (np.sum(seg_mask) / seg_mask.size) * 100
    if tumor_prob < 0.5:
        tumor_percentage = 0.0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="glass-card">
        <h3>🧠 Tumor Probability</h3>
        <h1>{tumor_prob*100:.2f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="glass-card">
        <h3>📊 Tumor Area</h3>
        <h1>{tumor_percentage:.2f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    if tumor_prob >= 0.5:
        diagnosis = "Tumor Detected"
    else:
        diagnosis = "Normal Liver"

    with col3:
        st.markdown(f"""
        <div class="glass-card">
        <h3>📌 Diagnosis</h3>
        <h1>{diagnosis}</h1>
        </div>
        """, unsafe_allow_html=True)

    st.progress(float(tumor_prob))

    # ----------------------------
    # Clinical Interpretation
    # ----------------------------
    st.markdown("""
    <div class="glass-card">
    <h3>🔬 Clinical Interpretation</h3>
    """, unsafe_allow_html=True)

    if tumor_prob >= 0.5:
        if tumor_percentage < 0.5:
            st.write("🟢 Mild Tumor Presence")
        elif tumor_percentage <= 2:
            st.write("🟡 Moderate Tumor Burden")
        else:
            st.write("🔴 Severe Tumor Burden")
    else:
        st.write("🟢 Liver Appears Normal")

    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------
    # Visualization
    # ----------------------------
    st.markdown("""
    <div class="glass-card">
    <h3>🖼️ Scan Visualization</h3>
    </div>
    """, unsafe_allow_html=True)

    slice_to_view = st.slider("Select Slice", 0, depth-1, center)
    show_overlay = st.checkbox("Show Tumor Overlay", value=True)

    colA, colB = st.columns(2)

    with colA:
        fig1, ax1 = plt.subplots()
        ax1.imshow(volume[:, :, slice_to_view], cmap="gray")
        ax1.axis("off")
        st.pyplot(fig1)

    with colB:
        fig2, ax2 = plt.subplots()
        ax2.imshow(volume[:, :, slice_to_view], cmap="gray")
        if show_overlay and tumor_prob >= 0.5:
            mid_slice_index = seq_len // 2
            ax2.imshow(seg_mask[mid_slice_index], alpha=0.5)
        ax2.axis("off")
        st.pyplot(fig2)
