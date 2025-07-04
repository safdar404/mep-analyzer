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

st.set_page_config(page_title="Alfanar MEP OCR Analyzer", layout="wide")

# --- Alfanar branding (logo + header) ---
logo_path = "alfanar-logo.png"  # Make sure this file is uploaded in the same folder
if os.path.exists(logo_path):
    logo_img = PILImage.open(logo_path)
    st.image(logo_img, width=150)
st.title("📐 Alfanar MEP Drawing Analyzer (OCR Powered)")

# --- Step 3: Multi-file uploader ---
uploaded_files = st.file_uploader(
    "Upload one or more MEP PDFs or Images", 
    type=["pdf", "png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

# Prepare to accumulate results from all files
all_data = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        ext = uploaded_file.name.split(".")[-1].lower()
        images = []
        if ext == "pdf":
            # Save temp PDF for conversion
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
        
        # For each page/image in the PDF or single image file
        for page_num, image in enumerate(images, start=1):
            st.markdown(f"### File: {uploaded_file.name} - Page {page_num}")
            st.image(image, use_column_width=True)

            # Preprocess for OCR: grayscale + resize + Otsu threshold
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            # Resize to improve OCR accuracy, limit max size to avoid memory error
            max_dim = 1000
            h, w = gray.shape
            scale = min(max_dim/w, max_dim/h, 1)  # scale down if larger than max_dim
            dim = (int(w * scale), int(h * scale))
            resized = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)

            _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # OCR with whitelist
            ocr_text = pytesseract.image_to_string(
                thresh,
                config='--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789x/L/S '
            )

            st.subheader("📝 Raw OCR Text")
            st.text_area(f"OCR Output for {uploaded_file.name} page {page_num}", ocr_text, height=200)

            # Extract info from OCR
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

            st.write("✅ Detected Labels:", detected_labels)
            st.write("💨 Airflows:", airflows)
            st.write("📏 Sizes:", sizes)

            # Summary for this page
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

    # Show combined summary table for all files and pages
    if all_data:
        df_all = pd.DataFrame(all_data)
        st.subheader("📊 Combined Summary from All Uploaded Files")
        edited_df = st.data_editor(df_all, use_container_width=True, num_rows="fixed")

        # Export combined summary to Excel
        towrite = io.BytesIO()
        edited_df.to_excel(towrite, index=False, sheet_name="MEP Summary")
        towrite.seek(0)
        st.download_button("📥 Download Combined Excel Summary", towrite,
                           "alfanar_mep_combined_summary.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.caption("© 2025 Alfanar MEP OCR Analyzer")
