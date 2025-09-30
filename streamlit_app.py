#!/usr/bin/env python3
"""
Streamlit NER Redaction App
- Loads a fine-tuned DistilBERT token classification model from ./model
- Takes a single text input, outputs redacted text replacing entities with placeholders like <firstname_01>
- Uses a single-user redaction log at ./redaction_logs/current_redaction.json (always overwritten)

Usage:
  streamlit run streamlit_app.py

Notes:
- Expects DistilBERT tokenizer/model plus label_mappings.json in the model directory.
- If label_mappings.json is missing, falls back to model.config.id2label.
- Only the latest redaction mapping is kept to ensure privacy and simplicity.
"""
import os
import json
import uuid
import datetime as dt
from typing import List, Dict, Any, Tuple

import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from huggingface_hub import hf_hub_download
import streamlit as st

# ---------------- Configuration ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Load model directly from Hugging Face repo
MODEL_DIR = "narayan214/distilbert-pii-before-v2"
LOGS_DIR = os.path.join(BASE_DIR, "redaction_logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "current_redaction.json")

MAX_LENGTH = 512  # adjust if your texts are very long
TAG_SCHEME = "BIO"  # model was trained with BIO in the provided training script
# ------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model_and_labels(model_dir: str):
    """Load tokenizer, model, label mappings."""
    # model_dir can be a local path or a Hugging Face repo id
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForTokenClassification.from_pretrained(model_dir)

    # Try to load explicit label mappings saved by training script
    id2label = None
    label2id = None
    # Try to load label mappings either locally or from the hub
    local_mapping_path = os.path.join(model_dir, 'label_mappings.json') if os.path.isdir(model_dir) else None
    if local_mapping_path and os.path.exists(local_mapping_path):
        try:
            with open(local_mapping_path, 'r', encoding='utf-8') as f:
                m = json.load(f)
                id2label = {int(k): v for k, v in m.get('id_to_label', {}).items()}
                label2id = {k: int(v) for k, v in m.get('label_to_id', {}).items()}
        except Exception:
            pass
    if id2label is None or label2id is None:
        try:
            hub_path = hf_hub_download(repo_id=model_dir, filename='label_mappings.json')
            with open(hub_path, 'r', encoding='utf-8') as f:
                m = json.load(f)
                id2label = {int(k): v for k, v in m.get('id_to_label', {}).items()}
                label2id = {k: int(v) for k, v in m.get('label_to_id', {}).items()}
        except Exception:
            pass

    # Fallback to model config if needed
    if id2label is None or not id2label:
        id2label = {int(k): v for k, v in getattr(model.config, 'id2label', {}).items()}
    if label2id is None or not label2id:
        label2id = {k: int(v) for k, v in getattr(model.config, 'label2id', {}).items()}

    # Normalize labels to uppercase types in BIO scheme (e.g., B-FIRSTNAME)
    # Keep as-is since model was trained with these exact strings.

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    return tokenizer, model, id2label, label2id, device


def decode_entities_bio(tokens_offsets: List[Tuple[int, int]], pred_labels: List[str]) -> List[Dict[str, Any]]:
    """Convert per-token BIO labels into contiguous entity spans in char space.

    Returns list of dicts: { 'start': int, 'end': int, 'entity_type': str }
    """
    spans = []
    current_type = None
    span_start = None
    span_end = None

    for i, (offset, label) in enumerate(zip(tokens_offsets, pred_labels)):
        start_c, end_c = offset
        if end_c <= start_c:  # special tokens or padding
            continue

        if label == 'O' or label is None:
            # close any open span
            if current_type is not None:
                spans.append({'start': span_start, 'end': span_end, 'entity_type': current_type})
                current_type = None
                span_start = None
                span_end = None
            continue

        # label like B-TYPE or I-TYPE
        if '-' in label:
            tag, etype = label.split('-', 1)
        else:
            tag, etype = 'O', None

        if tag == 'B':
            # close previous
            if current_type is not None:
                spans.append({'start': span_start, 'end': span_end, 'entity_type': current_type})
            current_type = etype
            span_start = start_c
            span_end = end_c
        elif tag == 'I' and current_type == etype and span_start is not None:
            # extend current
            span_end = end_c
        else:
            # invalid continuation -> start new
            if current_type is not None:
                spans.append({'start': span_start, 'end': span_end, 'entity_type': current_type})
            current_type = etype if etype else None
            span_start = start_c if etype else None
            span_end = end_c if etype else None

    # close tail
    if current_type is not None and span_start is not None:
        spans.append({'start': span_start, 'end': span_end, 'entity_type': current_type})

    # sort and merge tiny gaps (rare tokenizer artifacts)
    spans.sort(key=lambda x: (x['start'], x['end']))
    return spans


def infer_entities(text: str, tokenizer, model, id2label, device) -> List[Dict[str, Any]]:
    """Run NER inference and produce character-level entity spans with types and values."""
    # Tokenize with offsets
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=True,
    )
    input_ids = torch.tensor([enc['input_ids']], device=device)
    attention_mask = torch.tensor([enc['attention_mask']], device=device)
    offsets = enc['offset_mapping']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [1, seq_len, num_labels]
        preds = logits.argmax(dim=-1).squeeze(0).tolist()

    pred_labels = [id2label.get(int(i), 'O') for i in preds]

    # Filter out special tokens (offset (0,0) or tokenizer-specific)
    token_offsets = []
    filtered_labels = []
    for (s, e), lbl in zip(offsets, pred_labels):
        if e <= s:
            continue
        token_offsets.append((s, e))
        filtered_labels.append(lbl)

    spans = decode_entities_bio(token_offsets, filtered_labels)

    # attach extracted values
    for sp in spans:
        sp['value'] = text[sp['start']:sp['end']]
    return spans


def redact_text(text: str, spans: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Build a redacted string and mapping from original spans.

    Placeholder format: <entitytype_XX> where entitytype is lowercase, e.g. <firstname_01>
    Returns (redacted_text, mappings)
    mapping example entries: { 'placeholder': '<firstname_01>', 'entity_type': 'FIRSTNAME', 'value': 'Rajesh', 'start': 8, 'end': 14 }
    """
    # sort by start
    spans_sorted = sorted(spans, key=lambda x: (x['start'], x['end']))

    # merge overlapping spans by preferring longer span
    merged = []
    for sp in spans_sorted:
        if not merged or sp['start'] >= merged[-1]['end']:
            merged.append(sp)
        else:
            # overlap: keep the one with larger coverage
            prev = merged[-1]
            prev_len = prev['end'] - prev['start']
            sp_len = sp['end'] - sp['start']
            if sp_len > prev_len:
                merged[-1] = sp
            # else keep prev

    counters: Dict[str, int] = {}
    pieces = []
    last_idx = 0
    mappings = []

    for sp in merged:
        etype = sp['entity_type'] or 'ENTITY'
        etype_l = etype.lower()
        counters[etype_l] = counters.get(etype_l, 0) + 1
        placeholder = f"<{etype_l}_{counters[etype_l]:02d}>"

        # add preceding text
        if sp['start'] > last_idx:
            pieces.append(text[last_idx:sp['start']])
        # add placeholder
        pieces.append(placeholder)
        mappings.append({
            'placeholder': placeholder,
            'entity_type': etype,
            'value': sp.get('value', text[sp['start']:sp['end']]),
            'start': sp['start'],
            'end': sp['end'],
        })
        last_idx = sp['end']

    # tail
    if last_idx < len(text):
        pieces.append(text[last_idx:])

    redacted = ''.join(pieces)
    return redacted, mappings


def restore_text(redacted_text: str, mappings: List[Dict[str, Any]]) -> str:
    """Restore original text by replacing placeholders with their original values.

    Replaces all occurrences of each placeholder (e.g., <firstname_01>) with the
    corresponding original value from the mappings list.
    """
    if not mappings:
        return redacted_text

    restored = redacted_text
    # Replace placeholders; placeholders are unique tokens like <type_01>
    for m in mappings:
        placeholder = m.get('placeholder')
        value = m.get('value', '')
        if placeholder:
            restored = restored.replace(placeholder, value)
    return restored


def save_mapping(original_text: str, redacted_text: str, mappings: List[Dict[str, Any]]) -> str:
    """Save a reversible mapping JSON to the single-user log file.

    Always overwrites the previous log so only one mapping exists at a time.
    """
    rec = {
        'id': str(uuid.uuid4()),
        'created_at': dt.datetime.utcnow().isoformat() + 'Z',
        'original_text': original_text,
        'redacted_text': redacted_text,
        'mappings': mappings,
        'info': {
            'tag_scheme': TAG_SCHEME,
            'model_dir': MODEL_DIR,
            'max_length': MAX_LENGTH,
        }
    }
    # Ensure only one mapping is stored at a time
    if os.path.exists(LOG_FILE):
        try:
            os.remove(LOG_FILE)
        except Exception:
            # If removal fails for any reason, we will still attempt to overwrite
            pass
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return LOG_FILE


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="PII Redactor", page_icon="🛡️", layout="wide")

# Sidebar: Title, note, and sample selection
st.sidebar.title("🛡️ PII Redactor (DistilBERT)")
st.sidebar.info("This app keeps only the latest redaction mapping for a single user. Each new redaction deletes the previous mapping in `redaction_logs/current_redaction.json`.")

# Sidebar: Samples (Small >=30 words, Medium >=50 words, Long >60 words; Indian domain)
st.sidebar.markdown("**Samples**")
samples_by_size = {
    "Small": [
        (
            "Hello, my name is Rakesh Kumar from Pune, Maharashtra. You can reach me at 98230 12345 or "
            "rakesh.kumar@example.in. I visited HDFC Bank, JM Road branch, last week to update my "
            "Aadhaar-linked mobile number. The relationship manager also verified my PAN details during the visit."
        ),
        (
            "This is Priya Sharma in Bengaluru, Karnataka, working at Infosys. Her email is priya.sharma@infosys.co.in "
            "and phone 080-2659-1234. She met Dr. Mehta at Fortis Hospital to discuss insurance details on Tuesday "
            "morning around ten o'clock after confirming the appointment with the reception desk."
        ),
    ],
    "Medium": [
        (
            "Please contact Ananya Verma from Mumbai regarding the GST invoice correction for order number 78452. "
            "Her official email is ananya.verma@tcs.com and the mobile number is 98765 43210. Office address: Tata "
            "Consultancy Services, Nirmal Building, Nariman Point, Mumbai 400021. The review meeting with Mr. Agarwal "
            "is scheduled on 15 October 2025 at 11 AM, conference room two, and the finance team will join remotely."
        ),
        (
            "The student, Arjun Singh, enrolled at IIT Delhi, Hauz Khas, requires a bonafide certificate for visa processing. "
            "He can be contacted at arjun.singh@iitd.ac.in or 98100 11223. His permanent address is 17/4, Patel Nagar, New Delhi. "
            "The international office has asked him to submit his passport copy, Aadhaar, and fee receipt by Friday afternoon."
        ),
    ],
    "Long": [
        (
            "On 22 September 2025, at Apollo Hospitals, Jubilee Hills, Hyderabad, Mr. Arvind Nair consulted Dr. Kavita Rao for a "
            "follow-up regarding hypertension medication and routine blood tests. His contact number is 98765 43210 and email is "
            "arvind.nair@example.in. Billing desk recorded his address as Flat 502, Lakeview Residency, Road No. 10, Banjara Hills, "
            "Hyderabad 500034. The next appointment is planned for 10 October 2025 at 9:30 AM. Pharmacy advised switching to a "
            "generic brand available at MedPlus, Road No. 12, and the lab shared the fasting instructions via SMS earlier today."
        )
    ],
}
size_choice = st.sidebar.radio("Size", options=["Small", "Medium", "Long"], key="sample_size", horizontal=False)
sample_options = samples_by_size.get(size_choice, [])
sample_choice = st.sidebar.selectbox("Choose sample", options=["-- select --"] + sample_options, key="sample_choice")
if st.sidebar.button("Load sample", key="btn_load_sample"):
    if sample_choice and sample_choice != "-- select --":
        st.session_state["input_text"] = sample_choice

# Load resources
try:
    tokenizer, model, id2label, label2id, device = load_model_and_labels(MODEL_DIR)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

tab_redact, tab_restore = st.tabs(["Redact", "Restore"])

with tab_redact:
    st.subheader("Redact PII")

    # Main input textbox using session state to allow loading samples
    user_text = st.text_area(
        "Input Text",
        key="input_text",
        value=st.session_state.get("input_text", ""),
        height=260,
        placeholder="Paste text containing PII here...",
    )

    redacted_text_box = st.empty()

    if st.button("Redact", key="btn_redact"):
        if not st.session_state.get("input_text", "").strip():
            st.warning("Please enter some text to redact.")
        else:
            with st.spinner("Redacting..."):
                spans = infer_entities(user_text, tokenizer, model, id2label, device)
                redacted, mappings = redact_text(user_text, spans)
                log_path = save_mapping(user_text, redacted, mappings)
            redacted_text_box.text_area("Redacted Output", redacted, height=260)
            st.caption(f"Mapping saved to: {log_path}")

with tab_restore:
    st.subheader("Restore from Redacted")

    # Always load from the single-user log file
    mapping_data = None
    mapping_path = None
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                mapping_data = json.load(f)
            mapping_path = LOG_FILE
            st.success("Latest mapping loaded from the default log.")
        except Exception as e:
            st.error(f"Failed to read default log: {e}")
    else:
        st.info("No mapping found. Please run a redaction first.")

    redacted_input = st.text_area(
        "Redacted Text",
        value=(mapping_data.get("redacted_text", "") if mapping_data else ""),
        height=200,
        placeholder="Paste redacted text containing placeholders like <firstname_01>...",
    )

    if st.button("Restore", key="btn_restore"):
        if not mapping_data:
            st.warning("No mapping available. Please perform a redaction first.")
        elif not redacted_input.strip():
            st.warning("Please paste the redacted text to restore.")
        else:
            with st.spinner("Restoring..."):
                restored = restore_text(redacted_input, mapping_data.get("mappings", []))
            st.text_area("Restored Output", restored, height=200)
            if mapping_path:
                st.caption(f"Mapping source: {mapping_path}")
