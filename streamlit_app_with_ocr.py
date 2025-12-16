# """
# streamlit_app_with_ocr.py
# Streamlit UI that:
# - Accepts image uploads (png/jpg)
# - Runs OCR (pytesseract) to extract text
# - Sends extracted text to the MCQ generator
# """

# import streamlit as st
# from PIL import Image
# import pytesseract
# import io
# import textwrap
# import random

# # import your question generation function
# from qg_core import build_multi_correct_questions, get_wordnet_distractors

# # If Tesseract is installed in a non-standard location on Windows, set path here:
# # Example Windows path:
# # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# # Uncomment and change if needed:
# # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# st.set_page_config(page_title="Image → MCQ Generator", layout="wide")

# st.title("🖼️ → 📝 Image to MCQ Generator")
# st.write("Upload an image containing text (photo of a page, screenshot). OCR will extract the text and the app will generate MCQs (A–D, 1 correct).")

# with st.sidebar:
#     st.header("Settings")
#     max_questions = st.number_input("Max questions", min_value=1, max_value=30, value=8)
#     min_span_len = st.number_input("Minimum answer span length", min_value=2, max_value=20, value=4)
#     st.markdown("---")
#     show_answers = st.checkbox("Show correct answers", value=False)
#     st.markdown("---")
#     st.info("Tip: Clean, high-contrast images produce better OCR results.")

# uploaded_image = st.file_uploader("Upload image (png, jpg, jpeg)", type=["png", "jpg", "jpeg"])

# def ocr_image_to_text(image_pil, lang="eng"):
#     """
#     Extract text from PIL image using pytesseract.
#     lang: Tesseract language code (default 'eng').
#     """
#     try:
#         # Convert to grayscale to help OCR
#         gray = image_pil.convert("L")
#         # Optionally: apply thresholding / resizing if image is small
#         # text = pytesseract.image_to_string(gray, lang=lang)  # default
#         custom_config = r'--oem 3 --psm 6'  # good default: LSTM + assume a single uniform block of text
#         text = pytesseract.image_to_string(gray, config=custom_config, lang=lang)
#         return text
#     except Exception as e:
#         st.error(f"OCR failed: {e}")
#         return ""

# def convert_to_single_answer(options, correct_options):
#     """
#     Ensure exactly 1 correct answer and 4 options (A–D).
#     Reuses same function logic as the MCQ app.
#     """
#     if not correct_options:
#         correct = options[0] if options else "Answer"
#     else:
#         correct = correct_options[0]
#     opts = [correct]

#     seed_word = correct.split()[0] if correct else "concept"
#     wn = get_wordnet_distractors(seed_word)
#     distractor_pool = wn + ["None of the above", "Not applicable", "A related concept"]
#     for d in distractor_pool:
#         if len(opts) >= 4:
#             break
#         if d != correct and d not in opts:
#             opts.append(d)
#     while len(opts) < 4:
#         opts.append(f"Option {len(opts)+1}")
#     return opts, correct

# if uploaded_image is not None:
#     img_bytes = uploaded_image.read()
#     image = Image.open(io.BytesIO(img_bytes))
#     st.image(image, caption="Uploaded image", use_column_width=True)

#     st.markdown("## OCR Extracted Text (editable)")
#     extracted = ocr_image_to_text(image)
#     text_area_content = st.text_area("Edit extracted text if needed (improves Q quality):", value=extracted, height=300)

#     if st.button("Generate MCQs from image text") and text_area_content.strip():
#         with st.spinner("Generating MCQs from OCR text..."):
#             # Call your question generator (this returns multi-answer groups normally).
#             # We will convert each to single-correct MCQ using convert_to_single_answer.
#             raw_questions = build_multi_correct_questions(
#                 text_area_content,
#                 group_size=1,       # request 1 correct candidate group
#                 max_questions=max_questions,
#                 min_span_len=min_span_len
#             )

#         if not raw_questions:
#             st.warning("No questions generated — try editing the extracted text or increasing the allowed span length.")
#         else:
#             st.success(f"Generated {len(raw_questions)} questions from image text.")
#             for i, q in enumerate(raw_questions, start=1):
#                 st.markdown(f"### Q{i}. {q['question']}")
#                 cols = st.columns([2, 1])
#                 extracted_correct = [q["options"][ci] for ci in q.get("correct_indices", [])] if q.get("correct_indices") else []
#                 options, correct = convert_to_single_answer(q["options"], extracted_correct)
#                 seed = hash(q["question"]) & 0xffffffff
#                 rnd = random.Random(seed)
#                 rnd.shuffle(options)
#                 with cols[0]:
#                     for idx, opt in enumerate(options):
#                         letter = chr(65 + idx)
#                         if show_answers and opt == correct:
#                             st.markdown(f"- ✅ **{letter}. {opt}**")
#                         else:
#                             st.markdown(f"- {letter}. {opt}")
#                 with cols[1]:
#                     st.markdown("**Context sentence:**")
#                     st.write(textwrap.shorten(q.get("context_sentence", ""), width=200))

# else:
#     st.info("Upload an image to begin OCR.")




# # Put near the top of streamlit_app_with_ocr.py
# import pytesseract
# from PIL import Image
# import numpy as np

# # Optional: set explicit tesseract path on Windows (uncomment & adjust if required)
# # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# # Try to import easyocr but don't fail if it's not installed
# try:
#     import easyocr
#     _HAS_EASYOCR = True
# except Exception:
#     _HAS_EASYOCR = False

# def ocr_image_to_text(image_pil, lang="eng"):
#     """
#     Attempts OCR with pytesseract first. If that fails (Tesseract not installed),
#     falls back to easyocr if available. Returns extracted text (string).
#     """
#     # Try pytesseract
#     try:
#         gray = image_pil.convert("L")
#         custom_config = r'--oem 3 --psm 6'
#         text = pytesseract.image_to_string(gray, config=custom_config, lang=lang)
#         # if returned text is empty or minimal, we will consider fallback
#         if text and len(text.strip()) > 5:
#             return text
#     except Exception as e:
#         # print or log error; we'll try fallback
#         print("pytesseract OCR error:", e)

#     # Fallback: easyocr
#     if _HAS_EASYOCR:
#         try:
#             # convert PIL image to numpy array for easyocr
#             img_np = np.array(image_pil.convert("RGB"))
#             reader = easyocr.Reader([lang])  # e.g. ['en']
#             result = reader.readtext(img_np)
#             # join text segments
#             extracted = " ".join([seg[1] for seg in result])
#             if extracted and len(extracted.strip())>0:
#                 return extracted
#         except Exception as e:
#             print("easyocr OCR error:", e)

#     # If both fail, raise a clear error to show to user
#     raise RuntimeError("OCR failed: tesseract is not installed or it's not in your PATH, and easyocr fallback is not available. See README for installation.")

"============================================================================================"
# """
# streamlit_app_with_ocr.py

# Image -> MCQ Streamlit app using easyocr (CPU)
# Requirements (in venv):
#     pip install easyocr pillow opencv-python

# Place this file in the same folder as qg_core.py.
# Run:
#     streamlit run streamlit_app_with_ocr.py
# """

# import streamlit as st
# from PIL import Image
# import io
# import textwrap
# import random

# # We'll import qg_core for question generation (must be present)
# from qg_core import build_multi_correct_questions, get_wordnet_distractors

# # Try to import easyocr and numpy; if missing show helpful message to user
# try:
#     import easyocr
#     import numpy as np
#     _HAS_EASYOCR = True
# except Exception as e:
#     _HAS_EASYOCR = False
#     _EASYOCR_ERROR = str(e)

# st.set_page_config(page_title="Image → MCQ (EasyOCR)", layout="wide")
# st.title("🖼️ → 📝 Image to MCQ (EasyOCR)")
# st.write("Upload an image containing text (photo or screenshot). OCR (easyocr) will extract text, you can edit it, then generate MCQs (A–D, single correct).")

# with st.sidebar:
#     st.header("Settings")
#     max_questions = st.number_input("Max questions", min_value=1, max_value=30, value=8)
#     min_span_len = st.number_input("Minimum answer span length (chars)", min_value=2, max_value=20, value=4)
#     st.markdown("---")
#     show_answers = st.checkbox("Show correct answers", value=False)
#     st.markdown("---")
#     st.info("This app uses easyocr (Python-only). If easyocr isn't installed, follow the instructions below.")

# if not _HAS_EASYOCR:
#     st.error(
#         "easyocr or numpy is not installed in your environment.\n\n"
#         "Install with (inside your venv):\n\n"
#         "`pip install easyocr pillow opencv-python`\n\n"
#         "If pip fails due to torch, first install a compatible CPU torch wheel:\n\n"
#         "`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`\n\n"
#         "Then re-run:\n\n"
#         "`pip install easyocr pillow opencv-python`"
#     )
#     st.stop()

# # Create a single easyocr.Reader instance and cache it across runs
# @st.cache_resource
# def get_easyocr_reader(lang_list=None, gpu=False):
#     if lang_list is None:
#         lang_list = ["en"]
#     # easyocr.Reader downloads model files on first run and caches them in ~/.EasyOCR
#     return easyocr.Reader(lang_list, gpu=gpu)

# reader = get_easyocr_reader(lang_list=["en"], gpu=False)

# uploaded_image = st.file_uploader("Upload image (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])

# def ocr_image_to_text_easyocr(image_pil, langs=["en"]):
#     """
#     Run easyocr on a PIL Image and return extracted text (joined segments).
#     """
#     img_np = np.array(image_pil.convert("RGB"))
#     result = reader.readtext(img_np)  # returns list of (bbox, text, confidence)
#     # Join recognized text segments into paragraphs; preserve order returned by EasyOCR
#     texts = [seg[1] for seg in result if seg and len(seg) > 1]
#     # naive join with spaces — user can edit afterwards
#     extracted = " ".join(texts)
#     return extracted

# def convert_to_single_answer(options, correct_options):
#     """
#     Ensure exactly 1 correct answer and exactly 4 options (A–D).
#     - pick first correct if multiple
#     - fill distractors using WordNet-based candidates or safe fillers
#     """
#     if not options:
#         # fallback single trivial Q
#         return ["Option 1", "Option 2", "Option 3", "Option 4"], "Option 1"

#     # Choose one correct option (prefer any reported correct)
#     correct = None
#     if correct_options:
#         correct = correct_options[0]
#     else:
#         # pick the first option as correct if generator provided none
#         correct = options[0]

#     opts = [correct]

#     # Use wordnet distractors (seed on first word of correct answer)
#     seed_word = correct.split()[0] if correct else "concept"
#     wn = get_wordnet_distractors(seed_word)
#     fillers = wn + ["None of the above", "Not applicable", "A related concept", "Another option"]

#     for f in fillers:
#         if len(opts) >= 4:
#             break
#         if f not in opts:
#             opts.append(f)

#     # Final fill if still short
#     while len(opts) < 4:
#         opts.append(f"Option {len(opts)+1}")

#     return opts, correct

# if uploaded_image is not None:
#     img_bytes = uploaded_image.read()
#     image = Image.open(io.BytesIO(img_bytes))
#     # show image using width parameter (avoid deprecated use_column_width)
#     st.image(image, caption="Uploaded image", width=700)

#     st.markdown("## OCR Extracted Text (editable)")
#     try:
#         extracted = ocr_image_to_text_easyocr(image, langs=["en"])
#     except Exception as e:
#         st.error(f"OCR failed: {e}")
#         extracted = ""

#     text_area_content = st.text_area(
#         "Edit extracted text if needed (improves question quality).",
#         value=extracted,
#         height=300
#     )

#     if st.button("Generate MCQs from image text") and text_area_content.strip():
#         with st.spinner("Generating MCQs from extracted text..."):
#             # build questions (qg_core returns items with options and possible multiple corrects)
#             raw_questions = build_multi_correct_questions(
#                 text_area_content,
#                 group_size=1,        # we want single-correct candidates
#                 max_questions=max_questions,
#                 min_span_len=int(min_span_len)
#             )

#         if not raw_questions:
#             st.warning("No questions were generated. Try editing the extracted text or increasing allowed span length.")
#         else:
#             st.success(f"Generated {len(raw_questions)} questions from image text.")
#             for i, q in enumerate(raw_questions, start=1):
#                 st.markdown(f"### Q{i}. {q['question']}")
#                 cols = st.columns([2, 1])

#                 # Determine candidate correct options returned by generator (may be multiple)
#                 extracted_correct = [q["options"][ci] for ci in q.get("correct_indices", [])] if q.get("correct_indices") else []
#                 options, correct = convert_to_single_answer(q["options"], extracted_correct)

#                 # deterministic shuffle per-question
#                 seed = hash(q["question"]) & 0xffffffff
#                 rnd = random.Random(seed)
#                 rnd.shuffle(options)

#                 with cols[0]:
#                     for idx, opt in enumerate(options):
#                         letter = chr(65 + idx)  # A-D
#                         if show_answers and opt == correct:
#                             st.markdown(f"- ✅ **{letter}. {opt}**")
#                         else:
#                             st.markdown(f"- {letter}. {opt}")

#                 with cols[1]:
#                     st.markdown("**Context sentence:**")
#                     st.write(textwrap.shorten(q.get("context_sentence", ""), width=200))

# else:
#     st.info("Upload an image to begin (photo of page or screenshot).")



"""
streamlit_app_with_ocr.py

Image → OCR → MCQ Generator (Stable Version)

Requirements (inside venv):
pip install streamlit easyocr pillow opencv-python torch torchvision

Run:
streamlit run streamlit_app_with_ocr.py
"""

import streamlit as st
from PIL import Image
import io
import numpy as np
import random
import textwrap
import easyocr

# ---------------- OCR SETUP ----------------
@st.cache_resource
def load_easyocr():
    return easyocr.Reader(["en"], gpu=False)

reader = load_easyocr()

def ocr_image_to_text(image_pil):
    img_np = np.array(image_pil.convert("RGB"))
    result = reader.readtext(img_np)
    texts = [seg[1] for seg in result if len(seg) > 1]
    return " ".join(texts)


# ---------------- MCQ GENERATOR ----------------
def generate_mcqs_from_text(text, max_questions=10):
    """
    Robust MCQ generator for long OCR text
    Works even if text has no proper sentences.
    """
    clean_text = (
        text.replace("_", " ")
        .replace(",", " ")
        .replace("\n", " ")
    )

    words = [w for w in clean_text.split() if len(w) > 4]

    if len(words) < 6:
        return []

    random.shuffle(words)
    questions = []

    for i in range(min(max_questions, len(words))):
        answer = words[i]

        question = f"What is associated with '{answer}'?"

        distractors_pool = [w for w in words if w != answer]
        distractors = random.sample(
            distractors_pool,
            k=min(3, len(distractors_pool))
        )

        options = [answer] + distractors
        while len(options) < 4:
            options.append("Not applicable")

        random.shuffle(options)

        questions.append({
            "question": question,
            "options": options[:4],
            "correct": answer,
            "context": clean_text[:200]
        })

    return questions


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Image → MCQ Generator", layout="wide")
st.title("🖼️ Image → 📝 MCQ Generator")
st.write("Upload an image → Extract text → Generate Multiple Choice Questions")

with st.sidebar:
    st.header("Settings")
    max_questions = st.slider("Number of questions", 1, 20, 10)
    show_answers = st.checkbox("Show correct answers", False)

uploaded_image = st.file_uploader(
    "Upload image (jpg / png)", type=["jpg", "jpeg", "png"]
)

if uploaded_image:
    image = Image.open(io.BytesIO(uploaded_image.read()))
    st.image(image, caption="Uploaded Image", width=700)

    st.subheader("📄 OCR Extracted Text (Editable)")
    extracted_text = ocr_image_to_text(image)

    user_text = st.text_area(
        "Edit extracted text if needed",
        value=extracted_text,
        height=250
    )

    if st.button("Generate MCQs"):
        with st.spinner("Generating MCQs..."):
            mcqs = generate_mcqs_from_text(
                user_text,
                max_questions=max_questions
            )

        if not mcqs:
            st.warning("Not enough meaningful text to generate questions.")
        else:
            st.success(f"Generated {len(mcqs)} questions")

            for i, q in enumerate(mcqs, start=1):
                st.markdown(f"### Q{i}. {q['question']}")

                for idx, opt in enumerate(q["options"]):
                    letter = chr(65 + idx)
                    if show_answers and opt == q["correct"]:
                        st.markdown(f"- ✅ **{letter}. {opt}**")
                    else:
                        st.markdown(f"- {letter}. {opt}")

                with st.expander("Context"):
                    st.write(textwrap.fill(q["context"], 80))

else:
    st.info("👆 Upload an image to start generating MCQs")
