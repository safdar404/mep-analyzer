import streamlit as st
import pytesseract
import cv2
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from pdf2image import convert_from_path
import re
from collections import Counter
import io
import os

# ------------------------
# 🌐 PAGE CONFIG & STYLING
# ------------------------
st.set_page_config(page_title="Alfanar MEP OCR Analyzer", layout="wide", page_icon="📐")

# CSS to inject
st.markdown("""
    <style>
        .stApp {
            background-color: #f4f6f9;
        }
        .css-1d391kg {  # Title
            color: #1f4e79 !important;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ------------------------
# 🏷️ LOGO + TITLE
# ------------------------
logo_path = "alfanar-logo.png"
cols = st.columns([1, 4])
with cols[0]:
    if os.path.exists(logo_path):
        logo = PILImage.open(logo_path)
        st.image(logo, width=120)
with cols[1]:
    st.title("📐 Alfanar MEP Drawing Analyzer (OCR Powered)")

st.markdown("##### 📤 Upload your MEP PDFs or Drawings to extract components, airflow, and duct sizes using AI OCR.")

# ------------------------
# 📂 FILE UPLOAD SECTION
# ------------------------
uploaded_files = st.file_uploader(
    "📁 Upload one or more MEP Drawings (PDF / Image)", 
    type=["pdf", "png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

all_data = []

# ------------------------
# 🔍 PROCESS EACH FILE
# ------------------------
if uploaded_files:
    for uploaded_file in uploaded_files:
        ext = uploaded_file.name.split(".")[-1].lower()
        images = []

        # Convert PDF to images
        if ext == "pdf":
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                images = convert_from_path("temp.pdf", dpi=300)
            except Exception as e:
                st.error(f"❌ Failed to convert PDF {uploaded_file.name}: {e}")
                continue
        else:
            try:
                images = [PILImage.open(uploaded_file)]
            except Exception as e:
                st.error(f"❌ Failed to open image {uploaded_file.name}: {e}")
                continue

        # Process each page/image
        for page_num, image in enumerate(images, start=1):
            st.markdown(f"---\n### 📄 File: `{uploaded_file.name}` - Page {page_num}")

            st.image(image, use_column_width=True, caption="📷 Original Drawing")

            # --- OCR Preprocessing ---
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            # Resize (limit max to 1000px)
            h, w = gray.shape
            max_dim = 1000
            scale = min(max_dim / w, max_dim / h, 1)
            dim = (int(w * scale), int(h * scale))
            resized = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)

            _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # --- OCR Extraction ---
            ocr_text = pytesseract.image_to_string(
                thresh,
                config='--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789x/L/S '
            )

            st.text_area("📝 OCR Raw Output", ocr_text, height=150)

            # --- Parsing & Extraction ---
            text_lines = ocr_text.upper().splitlines()
            labels = ['SAD', 'RAD', 'EAD', 'FAD', 'FD', 'VCD']
            detected_labels = []
            airflows = []
            sizes = []

            for line in text_lines:
                for label in labels:
                    if label in line:
                        detected_labels.append(label)
                match_flow = re.search(r"(\d+)\s*L/S", line)
                if match_flow:
                    airflows.append(int(match_flow.group(1)))
                match_size = re.search(r"(\d{2,4})\s*[xX*]\s*(\d{2,4})", line)
                if match_size:
                    sizes.append(f"{match_size.group(1)}x{match_size.group(2)}")

            label_counts = Counter(detected_labels)
            avg_air = sum(airflows) // len(airflows) if airflows else 0
            common_size = sizes[0] if sizes else "N/A"

            for label in labels:
                all_data.append({
                    "File": uploaded_file.name,
                    "Page": page_num,
                    "Component": label,
                    "Count": label_counts[label],
                    "Average Airflow (L/s)": avg_air if label_counts[label] else 0,
                    "Common Size": common_size if label_counts[label] else "N/A"
                })

# ------------------------
# 📊 SUMMARY TABLE + EXPORT
# ------------------------
if all_data:
    df_all = pd.DataFrame(all_data)
    st.subheader("📊 Combined Component Summary")
    edited_df = st.data_editor(df_all, use_container_width=True, num_rows="dynamic")

    # Download as Excel
    towrite = io.BytesIO()
    edited_df.to_excel(towrite, index=False, sheet_name="MEP Summary")
    towrite.seek(0)

    st.download_button(
        label="📥 Download Summary as Excel",
        data=towrite,
        file_name="alfanar_mep_combined_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ------------------------
# 🔚 FOOTER
# ------------------------
st.markdown("---")
st.caption("🔧 Built for Alfanar | © 2025 • Developed by Muhammad Safdar")
