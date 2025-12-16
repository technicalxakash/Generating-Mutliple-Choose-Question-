# ocr_extractor.py
import sys
import argparse
import numpy as np
from PIL import Image, ImageOps

# ---------- TESSERACT SETUP ----------
import pytesseract

# 🔴 CHANGE THIS PATH IF YOUR TESSERACT IS INSTALLED ELSEWHERE
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------- OPTIONAL EASYOCR ----------
try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

# Create EasyOCR reader ONCE (important)
EASY_READER = easyocr.Reader(["en"], gpu=False) if HAS_EASYOCR else None


# ---------- IMAGE PREPROCESSING ----------
def preprocess_image(img_pil, bw_threshold=140):
    """
    Preprocess image for better OCR:
    - grayscale
    - auto contrast
    - resize
    - binary threshold
    """
    img = img_pil.convert("L")
    img = ImageOps.autocontrast(img)

    w, h = img.size
    if w < 2000:
        scale = 2000 / w
        img = img.resize((2000, int(h * scale)), Image.LANCZOS)

    img = img.point(lambda x: 0 if x < bw_threshold else 255, '1')
    return img


# ---------- OCR FUNCTIONS ----------
def ocr_with_tesseract(img_pil, lang="eng"):
    config = r"--oem 3 --psm 12"
    return pytesseract.image_to_string(img_pil, lang=lang, config=config)


def ocr_with_easyocr(img_pil):
    img_np = np.array(img_pil)
    results = EASY_READER.readtext(img_np)
    return "\n".join([r[1] for r in results])


# ---------- MAIN OCR PIPELINE ----------
def extract_text_from_image(path, lang="eng", out_file=None, bw_threshold=140):
    img = Image.open(path)
    img = preprocess_image(img, bw_threshold=bw_threshold)

    text = ""

    # 1️⃣ Try Tesseract
    try:
        text = ocr_with_tesseract(img, lang=lang)
        if text.strip():
            print("✔ OCR done using Tesseract")
    except Exception as e:
        print("⚠ Tesseract failed:", e)

    # 2️⃣ Fallback to EasyOCR
    if not text.strip() and HAS_EASYOCR:
        try:
            text = ocr_with_easyocr(img)
            print("✔ OCR done using EasyOCR")
        except Exception as e:
            print("⚠ EasyOCR failed:", e)

    if not text.strip():
        raise RuntimeError("OCR failed. Image quality too low or OCR not configured.")

    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(text)

    return text


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="OCR Text Extractor (Tesseract + EasyOCR fallback)")
    parser.add_argument("image", help="Path to image (jpg/png)")
    parser.add_argument("--lang", default="eng", help="Tesseract language code (eng, hin, etc.)")
    parser.add_argument("--out", help="Save extracted text to file")
    parser.add_argument("--bw-threshold", type=int, default=140, help="Binary threshold (0–255)")

    args = parser.parse_args()

    try:
        text = extract_text_from_image(
            args.image,
            lang=args.lang,
            out_file=args.out,
            bw_threshold=args.bw_threshold
        )
        print("\n----- OCR OUTPUT -----\n")
        print(text)
    except Exception as e:
        print("❌ ERROR:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
