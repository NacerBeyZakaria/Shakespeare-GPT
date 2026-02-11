import streamlit as st
import torch
from tokenizers import Tokenizer
from model import GPTModel
from config import *
import gc

# =====================================================
# Page config
# =====================================================
st.set_page_config(
    page_title="My Custom LLM Chat",
    page_icon="🤖",
    layout="wide"
)

# =====================================================
# Device
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# Load model & tokenizer (cached)
# =====================================================
@st.cache_resource
def load_model():
    tokenizer = Tokenizer.from_file("tokenizer.json")

    model = GPTModel(
        VOCAB_SIZE,
        EMBED_DIM,
        NUM_HEADS,
        NUM_LAYERS,
        MAX_SEQ_LEN
    )

    model.load_state_dict(
        torch.load("checkpoints/model.pth", map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    return model, tokenizer


# =====================================================
# Init
# =====================================================
try:
    model, tokenizer = load_model()
except FileNotFoundError:
    st.error(
        "Train the model first:\n\n"
        "`python train.py`\n\n"
        "Missing files:\n"
        "- checkpoints/model.pth\n"
        "- tokenizer.json"
    )
    st.stop()


# =====================================================
# Sidebar
# =====================================================
with st.sidebar:
    st.title("⚙️ Settings")
    temperature = st.slider("Temperature (creativity)", 0.1, 1.5, 0.8, 0.1)
    max_tokens = st.slider("Max Tokens", 50, 500, 150, 25)
    clear_btn = st.button("🗑️ Clear Chat", type="secondary")
    st.write(f"Device: `{DEVICE}`")


# =====================================================
# Chat state
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if clear_btn:
    st.session_state.messages = []
    st.rerun()


# =====================================================
# Display history
# =====================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =====================================================
# Chat input & generation
# =====================================================
if prompt := st.chat_input("Type your message here..."):

    # User message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant message
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        # Encode input
        context = torch.tensor(
            [tokenizer.encode(prompt).ids],
            dtype=torch.long,
            device=DEVICE
        )

        with torch.no_grad():
            for _ in range(max_tokens):

                logits = model(context)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)
                token_id = next_token.item()

                token_text = tokenizer.decode([token_id])
                full_response += token_text

                placeholder.markdown(full_response + "▌")

                # ✅ FIXED: no unsqueeze
                context = torch.cat([context, next_token], dim=1)

                if token_id == tokenizer.token_to_id("<eos>"):
                    break

        placeholder.markdown(full_response)

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    st.rerun()
