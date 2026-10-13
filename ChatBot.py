"""
RAG Based YouTube ChatBot  -  Streamlit front end
=================================================
Pipeline (same as the notebook RAG_Using_LangChain.ipynb):
  1. Fetch transcript   -> youtube_transcript_api
  2. Split into chunks  -> RecursiveCharacterTextSplitter
  3. Embed + store      -> OpenAIEmbeddings + FAISS
  4. Retrieve + Augment -> retriever | prompt
  5. Generate           -> ChatOpenAI (streamed to the chat UI)

Run:
    pip install streamlit langchain langchain-openai langchain-community \
                langchain-text-splitters faiss-cpu youtube-transcript-api
    export OPENAI_API_KEY="sk-..."        # or paste it in the sidebar
    streamlit run Rag_Based_Youtube_ChatBot.py
"""

import html
import os
import re

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    AgeRestricted,
    InvalidVideoId,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)

# --------------------------------------------------------------------------- #
# Page config + styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="RAG YouTube ChatBot",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="expanded",
)

YT_LOGO_SVG = """
<svg viewBox="0 0 159 110" xmlns="http://www.w3.org/2000/svg" width="{w}" aria-label="YouTube">
  <path fill="#FF0000" d="M154 17.5c-1.8-6.7-7.1-12-13.8-13.8C128 .4 79.5.4 79.5.4S31 .4 18.8 3.7C12.1 5.5 6.8 10.8 5 17.5 1.7 29.7 1.7 55 1.7 55s0 25.3 3.3 37.5c1.8 6.7 7.1 12 13.8 13.8 12.2 3.3 60.7 3.3 60.7 3.3s48.5 0 60.7-3.3c6.7-1.8 12-7.1 13.8-13.8 3.3-12.2 3.3-37.5 3.3-37.5s0-25.3-3.3-37.5z"/>
  <path fill="#fff" d="M64 78.6 104.3 55 64 31.4z"/>
</svg>
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.stApp {
    background:
      radial-gradient(900px 500px at 8% -10%, rgba(255,0,0,.20), transparent 60%),
      radial-gradient(800px 500px at 100% 0%, rgba(120,60,255,.18), transparent 60%),
      #0b0b12;
    color: #ececf3;
}
/* Keep the header (it holds the sidebar open/close button) but make it invisible */
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { visibility: hidden; }
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"] { visibility: visible !important; color: #fff !important; }
.block-container { padding-top: 2rem; max-width: 1100px; }

/* Hero */
.hero { text-align: center; padding: 1.2rem 0 .4rem; }
.hero .logo { display: inline-block; filter: drop-shadow(0 8px 28px rgba(255,0,0,.45)); }
.hero h1 {
    font-size: 2.7rem; font-weight: 800; margin: .5rem 0 .2rem; letter-spacing: -.02em;
    background: linear-gradient(90deg, #fff 0%, #ff6b6b 60%, #ff1f1f 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { color: #a9a9bd; font-size: 1.05rem; max-width: 720px; margin: 0 auto; line-height: 1.6; }
.badges { margin-top: 1rem; display: flex; gap: .5rem; justify-content: center; flex-wrap: wrap; }
.badge {
    font-size: .78rem; padding: .3rem .75rem; border-radius: 999px;
    background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12); color: #d5d5e4;
}

/* Cards */
.card {
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.09);
    border-radius: 18px; padding: 1.2rem 1.4rem; backdrop-filter: blur(8px);
}
.step { display: flex; align-items: center; gap: .6rem; margin: 1.6rem 0 .7rem; }
.step .num {
    width: 30px; height: 30px; border-radius: 50%; background: #ff1f1f; color: #fff;
    display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: .9rem;
    box-shadow: 0 0 18px rgba(255,31,31,.5);
}
.step .title { font-size: 1.25rem; font-weight: 700; }

/* Inputs / buttons */
.stTextInput input {
    background: rgba(255,255,255,.06) !important; color: #fff !important;
    border: 1px solid rgba(255,255,255,.15) !important; border-radius: 12px !important;
    padding: .75rem 1rem !important;
}
.stTextInput input:focus { border-color: #ff3b3b !important; box-shadow: 0 0 0 3px rgba(255,59,59,.25) !important; }
.stFormSubmitButton button, .stDownloadButton button, .stButton button {
    background: linear-gradient(135deg, #ff1f1f, #c40000) !important; color: #fff !important;
    border: 0 !important; border-radius: 12px !important; font-weight: 600 !important;
    padding: .65rem 1.2rem !important; transition: transform .15s, box-shadow .15s;
}
.stFormSubmitButton button:hover, .stDownloadButton button:hover, .stButton button:hover {
    transform: translateY(-1px); box-shadow: 0 8px 24px rgba(255,31,31,.4);
}
[data-testid="stForm"] { border: 0 !important; padding: 0 !important; background: transparent !important; }

/* Transcript */
.transcript {
    max-height: 380px; overflow-y: auto; line-height: 1.75; font-size: .96rem; color: #dcdcea;
    padding-right: .6rem; white-space: pre-wrap;
}
.transcript::-webkit-scrollbar { width: 8px; }
.transcript::-webkit-scrollbar-thumb { background: rgba(255,255,255,.2); border-radius: 8px; }
.ts { color: #ff6b6b; font-weight: 600; font-variant-numeric: tabular-nums; margin-right: .35rem; }
.stat { display:inline-block; margin-right: 1.2rem; color:#a9a9bd; font-size:.85rem; }
.stat b { color:#fff; }

/* Chat */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.08);
    border-radius: 16px; padding: .9rem 1.1rem; margin-bottom: .6rem;
}
[data-testid="stChatInput"] { border-radius: 16px; }
[data-testid="stSidebar"] { background: #0f0f18; border-right: 1px solid rgba(255,255,255,.07); }
.small { color:#8d8da3; font-size:.82rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
PROMPT = PromptTemplate(
    template=(
        "You are a helpful assistant answering questions about a YouTube video.\n"
        "Answer ONLY from the provided transcript context.\n"
        "If the context is insufficient, just say you don't know.\n\n"
        "Transcript context:\n{context}\n\n"
        "Question: {question}\n\nAnswer:"
    ),
    input_variables=["context", "question"],
)

ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(text: str):
    """Accept a bare 11-char ID or any common YouTube URL; return the ID or None."""
    text = (text or "").strip()
    if ID_RE.match(text):
        return text
    m = re.search(r"(?:v=|youtu\.be/|/embed/|/shorts/|/live/)([A-Za-z0-9_-]{11})", text)
    return m.group(1) if m else None


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fetch_transcript(video_id: str):
    """Return (list_of_snippets, language_code). Prefers English, falls back to any language."""
    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(video_id, languages=["en", "en-US", "en-GB"])
    except NoTranscriptFound:
        # No English captions: use whatever transcript exists.
        transcript = next(iter(api.list(video_id)))
        fetched = transcript.fetch()
    snippets = [
        {"text": s.text.replace("\n", " "), "start": s.start} for s in fetched
    ]
    return snippets, getattr(fetched, "language_code", "")


def build_index(transcript: str, api_key: str):
    """Split -> embed -> FAISS. Returns (vector_store, n_chunks)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = splitter.create_documents([transcript])
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    return FAISS.from_documents(chunks, embeddings), len(chunks)


def build_chain(vector_store, api_key: str, model: str, temperature: float, k: int):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key, streaming=True)
    format_docs = RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs))
    parallel = RunnableParallel(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    )
    return parallel | PROMPT | llm | StrOutputParser()


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, TranscriptsDisabled):
        return "Subtitles/captions are disabled for this video, so there is no transcript to read."
    if isinstance(exc, NoTranscriptFound):
        return "No transcript was found for this video."
    if isinstance(exc, VideoUnavailable):
        return "This video is unavailable (private, removed, or the ID is wrong)."
    if isinstance(exc, AgeRestricted):
        return "This video is age-restricted, so its transcript can't be fetched."
    if isinstance(exc, InvalidVideoId):
        return "That doesn't look like a valid YouTube video ID."
    if isinstance(exc, (IpBlocked, RequestBlocked)):
        return (
            "YouTube is blocking requests from this network/IP (common on cloud servers). "
            "Run the app on your own computer or configure a proxy."
        )
    return f"Something went wrong: {exc}"


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
for key, default in {
    "video_id": None,
    "snippets": [],
    "transcript": "",
    "lang": "",
    "vector_store": None,
    "n_chunks": 0,
    "messages": [],
}.items():
    st.session_state.setdefault(key, default)


def default_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            key = ""
    return key


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(YT_LOGO_SVG.format(w=70), unsafe_allow_html=True)
    st.markdown("### Settings")
    api_key = st.text_input(
        "OpenAI API key",
        value=default_api_key(),
        type="password",
        help="Read from OPENAI_API_KEY / Streamlit secrets if set; otherwise paste it here.",
        placeholder="sk-...",
    )
    model = st.selectbox("Chat model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
    top_k = st.slider("Chunks retrieved (k)", 1, 8, 4)
    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown(
        "<p class='small'>How it works: transcript → chunks → OpenAI embeddings → "
        "FAISS vector store → retrieve top-k → GPT answers only from the video.</p>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
<div class="hero">
  <div class="logo">{YT_LOGO_SVG.format(w=110)}</div>
  <h1>RAG YouTube ChatBot</h1>
  <p>Paste any YouTube video ID and chat with the video. The bot fetches the transcript,
  turns it into embeddings stored in a FAISS vector database, and answers your questions
  using only what was actually said in the video.</p>
  <div class="badges">
    <span class="badge">🦜 LangChain</span><span class="badge">🧠 OpenAI Embeddings</span>
    <span class="badge">📦 FAISS</span><span class="badge">⚡ Streamlit</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Step 1 - video ID
# --------------------------------------------------------------------------- #
st.markdown(
    "<div class='step'><div class='num'>1</div><div class='title'>Enter a YouTube video ID</div></div>",
    unsafe_allow_html=True,
)
with st.form("video_form", clear_on_submit=False):
    c1, c2 = st.columns([5, 1.2])
    vid_input = c1.text_input(
        "Video ID",
        placeholder="e.g. 0zFfcEr1e9U   (a full YouTube link works too)",
        label_visibility="collapsed",
    )
    submitted = c2.form_submit_button("Load video", use_container_width=True)

if submitted:
    vid = extract_video_id(vid_input)
    if not vid:
        st.error("Please enter a valid 11-character YouTube video ID (e.g. 0zFfcEr1e9U) or a YouTube link.")
    elif not api_key:
        st.error("Please add your OpenAI API key in the sidebar (click the » arrow at the top-left "
                 "to open it), or set the OPENAI_API_KEY environment variable.")
    elif vid == st.session_state.video_id and st.session_state.vector_store is not None:
        st.info("This video is already loaded - ask away below!")
    else:
        try:
            with st.status("Processing video...", expanded=True) as status:
                st.write("📥 Fetching transcript from YouTube...")
                snippets, lang = fetch_transcript(vid)
                transcript = " ".join(s["text"] for s in snippets).strip()
                if not transcript:
                    raise ValueError("The transcript is empty.")
                st.write(f"✂️ Splitting {len(transcript):,} characters and creating embeddings...")
                store, n_chunks = build_index(transcript, api_key)
                status.update(label="Video ready!", state="complete", expanded=False)
            st.session_state.update(
                video_id=vid,
                snippets=snippets,
                transcript=transcript,
                lang=lang,
                vector_store=store,
                n_chunks=n_chunks,
                messages=[],  # new video -> fresh conversation
            )
        except Exception as exc:  # noqa: BLE001 - show a friendly message in the UI
            st.error(friendly_error(exc))

# --------------------------------------------------------------------------- #
# Step 2 - transcript
# --------------------------------------------------------------------------- #
if st.session_state.vector_store is not None:
    vid = st.session_state.video_id
    st.markdown(
        "<div class='step'><div class='num'>2</div><div class='title'>Video transcript</div></div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.markdown(
            f"<div class='card'><img src='https://img.youtube.com/vi/{vid}/hqdefault.jpg' "
            f"style='width:100%;border-radius:12px;margin-bottom:.8rem'/>"
            f"<span class='stat'>Characters <b>{len(st.session_state.transcript):,}</b></span>"
            f"<span class='stat'>Words <b>{len(st.session_state.transcript.split()):,}</b></span><br>"
            f"<span class='stat'>Chunks <b>{st.session_state.n_chunks}</b></span>"
            f"<span class='stat'>Language <b>{html.escape(st.session_state.lang or 'n/a')}</b></span><br>"
            f"<a href='https://www.youtube.com/watch?v={vid}' target='_blank' "
            f"style='color:#ff6b6b'>Open on YouTube ↗</a></div>",
            unsafe_allow_html=True,
        )
    with right:
        show_ts = st.toggle("Show timestamps", value=False)
        if show_ts:
            body = "<br>".join(
                f"<span class='ts'>[{fmt_time(s['start'])}]</span>{html.escape(s['text'])}"
                for s in st.session_state.snippets
            )
        else:
            body = html.escape(st.session_state.transcript)
        st.markdown(f"<div class='card transcript'>{body}</div>", unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download transcript",
            st.session_state.transcript,
            file_name=f"{vid}_transcript.txt",
        )

    # ----------------------------------------------------------------------- #
    # Step 3 - chat
    # ----------------------------------------------------------------------- #
    st.markdown(
        "<div class='step'><div class='num'>3</div><div class='title'>Ask questions about the video</div></div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.markdown(
            "<p class='small'>Try: “Can you summarize the video?”, “What are the key points?”, "
            "or ask about any specific topic mentioned.</p>",
            unsafe_allow_html=True,
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    question = st.chat_input("Ask something about this video...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🤖"):
            if not api_key:
                answer = "Please add your OpenAI API key in the sidebar."
                st.error(answer)
            else:
                try:
                    chain = build_chain(
                        st.session_state.vector_store, api_key, model, temperature, top_k
                    )
                    answer = st.write_stream(chain.stream(question))
                except Exception as exc:  # noqa: BLE001
                    answer = f"⚠️ Could not generate an answer: {exc}"
                    st.error(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.markdown(
        "<div class='card' style='margin-top:1.5rem;text-align:center;color:#a9a9bd'>"
        "👆 Enter a video ID above and press <b>Enter</b> or <b>Load video</b> to begin."
        "<br><span class='small'>The video needs captions (auto-generated captions work too).</span></div>",
        unsafe_allow_html=True,
    )
