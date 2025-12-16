
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import spacy
import nltk
from nltk.corpus import wordnet as wn
from sklearn.cluster import AgglomerativeClustering
import random
import re

# First-time NLTK downloads
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Load spaCy model (ensure en_core_web_sm installed separately)
nlp = spacy.load("en_core_web_sm")

# You can change to a different QG model if you prefer
QG_MODEL = "valhalla/t5-small-qg-hl"
tokenizer = AutoTokenizer.from_pretrained(QG_MODEL)
qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_MODEL)

# Sentence-transformer for semantic embeddings
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def split_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 8]

def extract_answer_spans(text, min_len=3):
    """
    Extract named entities and noun chunks as answer candidates.
    Returns list of tuples (text, start_char, end_char, sentence_text).
    """
    doc = nlp(text)
    spans = []
    for ent in doc.ents:
        spans.append((ent.text.strip(), ent.start_char, ent.end_char, ent.sent.text.strip()))
    for chunk in doc.noun_chunks:
        txt = chunk.text.strip()
        if len(txt) >= min_len and not txt.lower().startswith("the "):
            spans.append((txt, chunk.start_char, chunk.end_char, chunk.root.sent.text.strip()))
    # deduplicate preserving order
    seen = set()
    uniq = []
    for s in spans:
        key = re.sub(r'\s+',' ', s[0].lower())
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq

def format_for_t5_highlight(sentence, answer_span):
    pattern = re.escape(answer_span)
    highlighted = re.sub(pattern, "<hl> " + answer_span + " <hl>", sentence, count=1)
    return "generate question: " + highlighted

def generate_question_for_span(sentence, span_text, max_len=64):
    inp = format_for_t5_highlight(sentence, span_text)
    inputs = tokenizer(inp, return_tensors="pt", truncation=True)
    outputs = qg_model.generate(**inputs, max_length=max_len, num_beams=4, early_stopping=True)
    q = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return q

def get_semantic_neighbors(candidate, pool, top_k=5):
    """
    Return list of (text, score) from pool most similar to candidate.
    """
    if len(pool) == 0:
        return []
    sentences = [candidate] + pool
    emb = embed_model.encode(sentences, convert_to_tensor=True)
    query = emb[0:1]
    corpus = emb[1:]
    hits = util.semantic_search(query, corpus, top_k=top_k)[0]
    results = []
    for hit in hits:
        idx = hit['corpus_id']
        score = hit['score']
        results.append((pool[idx], float(score)))
    return results

def get_wordnet_distractors(word):
    distractors = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_',' ')
            if name.lower() != word.lower():
                distractors.add(name)
    return list(distractors)[:5]

def build_multi_correct_questions(chapter_text, group_size=2, max_questions=10, min_span_len=3):
    """
    Returns list of question dicts:
    {
      "question": str,
      "options": [str,...],
      "correct_indices": [int,...],
      "context_sentence": str
    }
    """
    sentences = split_sentences(chapter_text)
    spans = extract_answer_spans(chapter_text, min_len=min_span_len)
    span_texts = [s[0] for s in spans if len(s[0]) >= min_span_len]
    if len(span_texts) < 2:
        return []

    # Embeddings & clustering to form groups of related answers
    embeddings = embed_model.encode(span_texts, convert_to_tensor=True)
    n_clusters = max(1, min(len(span_texts) // group_size, 12))
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(embeddings.cpu().numpy())
    clusters = {}
    for idx, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(span_texts[idx])

    questions = []
    for cluster_spans in clusters.values():
        if len(cluster_spans) < group_size:
            continue
        # form groups (sliding windows) of size group_size
        for i in range(0, len(cluster_spans), group_size):
            chosen = cluster_spans[i:i+group_size]
            if len(chosen) < group_size:
                continue
            # choose the best context sentence
            best_sentence = None
            best_count = 0
            for sent in sentences:
                count = sum(1 for sp in chosen if sp.lower() in sent.lower())
                if count > best_count:
                    best_count = count
                    best_sentence = sent
            if not best_sentence:
                best_sentence = sentences[0]

            # Generate stem (highlight first answer)
            try:
                q_stem = generate_question_for_span(best_sentence, chosen[0])
            except Exception as e:
                # fallback simple stem
                q_stem = f"Which of the following are related to '{chosen[0]}'?"

            # assemble options
            options = []
            correct_indices = []
            for c in chosen:
                options.append(c)
            pool = [s for s in span_texts if s not in chosen]
            neighbors = get_semantic_neighbors(chosen[0], pool, top_k=8)
            distractors_added = 0
            for cand, score in neighbors:
                if any(cand.lower() in x.lower() or x.lower() in cand.lower() for x in chosen):
                    continue
                options.append(cand)
                distractors_added += 1
                if distractors_added >= 4:
                    break
            if distractors_added < 2:
                wn_d = get_wordnet_distractors(chosen[0].split()[0])
                for d in wn_d:
                    if d not in options:
                        options.append(d)
                        distractors_added += 1
                        if distractors_added >= 4:
                            break

            # Shuffle options
            combined = options[:]
            random.shuffle(combined)
            new_correct = [idx for idx,opt in enumerate(combined) if opt in chosen]

            question_obj = {
                "question": (q_stem + " (Select all that apply.)").strip(),
                "options": combined,
                "correct_indices": new_correct,
                "context_sentence": best_sentence
            }
            questions.append(question_obj)
            if len(questions) >= max_questions:
                break
        if len(questions) >= max_questions:
            break

    return questions
