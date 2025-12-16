"""
streamlit_app.py
MCQ Version — EXACTLY 1 Correct Answer per Question
- Always 4 options (A–D)
- Only 1 correct option
- Answers hidden by default
"""

import streamlit as st
from qg_core import build_multi_correct_questions, get_wordnet_distractors
import textwrap
import random

st.set_page_config(page_title="MCQ Generator", layout="wide")

st.title("🎯 MCQ Generator (1 Correct Answer Only)")
st.write("Paste or upload a chapter. This app generates **standard MCQs** with exactly **4 options (A–D)** and **only ONE correct answer**.")

# SIDEBAR
with st.sidebar:
    st.header("Settings")
    st.markdown("### Correct answers per question")
    st.write("✔ Fixed to **1** (MCQ mode)")
    correct_answers_per_question = 1  # fixed
    
    max_questions = st.number_input("Max questions", min_value=1, max_value=30, value=8)
    min_span_len = st.number_input("Minimum answer span length", min_value=2, max_value=20, value=3)

    st.markdown("---")
    show_answers = st.checkbox("Show correct answers", value=False)
    st.markdown("---")

uploaded_file = st.file_uploader("Upload chapter (.txt)", type=["txt"])
if uploaded_file is not None:
    chapter_text = uploaded_file.read().decode("utf-8")
else:
    chapter_text = st.text_area("Or paste chapter text here", height=300)


# FUNCTION TO FORCE 1 CORRECT ANSWER + 4 OPTIONS
def convert_to_single_answer(options, correct_options):
    """
    Ensures:
    - exactly 1 correct answer (choose the first)
    - total options = 4
    """
    if not correct_options:
        correct = options[0]
    else:
        correct = correct_options[0]  # pick only ONE correct answer

    opts = [correct]

    # Fill remaining slots with distractors
    seed_word = correct.split()[0]
    wn = get_wordnet_distractors(seed_word)

    distractor_pool = wn + ["None of the above", "Not applicable", "A related concept"]

    for d in distractor_pool:
        if len(opts) >= 4:
            break
        if d != correct and d not in opts:
            opts.append(d)

    # If still fewer than 4, fill generically
    while len(opts) < 4:
        opts.append(f"Option {len(opts)+1}")

    return opts, correct


# GENERATE QUESTIONS
if st.button("Generate Questions") and chapter_text.strip():
    with st.spinner("Generating Questions…"):
        mcq_list = build_multi_correct_questions(
            chapter_text,
            group_size=1,        # FORCE 1 correct answer internally
            max_questions=max_questions,
            min_span_len=min_span_len
        )

    if not mcq_list:
        st.warning("Could not generate questions. Try a larger chapter.")
    else:
        st.success(f"Generated {len(mcq_list)} MCQs")

        for i, q in enumerate(mcq_list, start=1):
            st.markdown(f"### Q{i}. {q['question']}")
            cols = st.columns([2, 1])

            # Extract correct answer → only ONE
            extracted_correct = [q["options"][ci] for ci in q["correct_indices"]]
            options, correct = convert_to_single_answer(q["options"], extracted_correct)

            # Shuffle (but consistent for same question)
            seed = hash(q["question"]) & 0xffffffff
            rnd = random.Random(seed)
            rnd.shuffle(options)

            with cols[0]:
                for idx, opt in enumerate(options):
                    letter = chr(65 + idx)

                    if show_answers and opt == correct:
                        st.markdown(f"- ✅ **{letter}. {opt}**")
                    else:
                        st.markdown(f"- {letter}. {opt}")

            with cols[1]:
                st.markdown("**Context sentence:**")
                st.write(textwrap.shorten(q["context_sentence"], width=200))

else:
    st.info("Upload or paste a chapter to begin.")
