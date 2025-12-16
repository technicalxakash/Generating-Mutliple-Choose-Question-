# ocr_streamlit.py
import streamlit as st
from PIL import Image
import io
import textwrap
import random
import numpy as np




# Local OCR extractor function reuse (you can import from ocr_extractor if you prefer)
try:
    import pytesseract
    _HAS_PYTESSERACT = True
except Exception:
    _HAS_PYTESSERACT = False

try:
    import easyocr
    _HAS_EASYOCR = True
except Exception:
    _HAS_EASYOCR = False

st.set_page_config(page_title="Image OCR Extractor", layout="wide")
st.title("Image → Text (OCR Extractor)")
st.write("Upload an image and extract editable text. Uses pytesseract (if installed) and falls back to easyocr.")

with st.sidebar:
    st.header("Settings")
    lang = st.text_input("Language code for OCR", value="en")
    show_download = st.checkbox("Show Download button", value=True)
    st.markdown("---")
    st.info("If you don't have system Tesseract, install easyocr: pip install easyocr")

uploaded = st.file_uploader("Upload image (png/jpg)", type=["png", "jpg", "jpeg"])
if uploaded:
    image = Image.open(io.BytesIO(uploaded.read()))
    st.image(image, width=700)

    # Preprocess: convert to grayscale & small resize (optional)
    def preprocess(img_pil):
        gray = img_pil.convert("RGB")
        w,h = gray.size
        if w < 1000:
            gray = gray.resize((1200, int(1200*h/w)))
        return gray

    img_proc = preprocess(image)

    def ocr_with_tesseract(img_pil, lang):
        config = r'--oem 3 --psm 6'
        return pytesseract.image_to_string(img_pil, lang=lang, config=config)

    def ocr_with_easyocr(img_pil, langs):
        reader = easyocr.Reader(langs, gpu=False)
        arr = np.array(img_pil)
        res = reader.readtext(arr)
        return "\n".join([seg[1] for seg in res])

    extracted = ""
    # try pytesseract
    if _HAS_PYTESSERACT:
        try:
            extracted = ocr_with_tesseract(img_proc, lang if len(lang)>0 else "eng")
        except Exception:
            extracted = ""

    if (not extracted or len(extracted.strip())<5) and _HAS_EASYOCR:
        try:
            extracted = ocr_with_easyocr(img_proc, [lang])
        except Exception:
            extracted = ""

    if not extracted:
        st.error("OCR failed. Install Tesseract (system) or easyocr (pip).")
    else:
        st.subheader("OCR Extracted Text (edit if needed)")
        txt = st.text_area("Edit extracted text before saving or using", value=extracted, height=300)
        if show_download and st.button("Download text"):
            b = io.BytesIO(txt.encode("utf-8"))
            st.download_button("Download .txt", b, file_name="extracted_text.txt")
