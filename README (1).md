# ▶️ RAG YouTube ChatBot

Chat with any YouTube video. Enter a video ID, and the app fetches the transcript, indexes it in a vector database, and answers your questions using **only what was actually said in the video**.

Built with **LangChain**, **OpenAI**, **FAISS** and **Streamlit**.

---

## ✨ Features

- 🎬 Enter a YouTube video ID (a full YouTube link works too)
- 📜 View the full transcript, with optional timestamps, and download it as a `.txt` file
- 💬 Ask questions in a modern chat interface with streaming answers
- 🧠 Retrieval-Augmented Generation (RAG): answers come only from the video's transcript, and the bot says "I don't know" when the answer isn't there
- ⚙️ Adjustable settings: chat model, temperature, and number of retrieved chunks
- 🎨 Clean dark UI with YouTube branding

## 🧩 How It Works

```
YouTube Video ID
      │
      ▼
 Fetch transcript        (youtube-transcript-api)
      │
      ▼
 Split into chunks       (RecursiveCharacterTextSplitter)
      │
      ▼
 Create embeddings       (OpenAI text-embedding-3-small)
      │
      ▼
 Store in vector DB      (FAISS)
      │
 User question ──► Retrieve top-k relevant chunks
                         │
                         ▼
               Prompt + context ──► OpenAI GPT ──► Answer
```

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| Orchestration | LangChain |
| Embeddings and LLM | OpenAI |
| Vector store | FAISS |
| Transcripts | youtube-transcript-api |

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install dependencies

```bash
pip install streamlit langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu youtube-transcript-api
```

### 3. Set your OpenAI API key

**Windows (PowerShell)**
```powershell
$env:OPENAI_API_KEY="sk-..."
```

**macOS / Linux**
```bash
export OPENAI_API_KEY="sk-..."
```

You can also paste the key into the app's sidebar. Never commit your API key to GitHub.

### 4. Run the app

```bash
streamlit run Rag_Based_Youtube_ChatBot.py
```

The app opens at `http://localhost:8501`.

## 📖 Usage

1. Enter a video ID such as `0zFfcEr1e9U` and press **Enter** or click **Load video**.
2. Read the transcript, or download it.
3. Ask questions about the video in the chat, for example "Can you summarize the video?"

> **Note:** The video must have captions (auto-generated captions work). YouTube may block transcript requests from cloud servers, so running the app locally works best.

## 📁 Project Structure

```
├── Rag_Based_Youtube_ChatBot.py   # Streamlit app (UI + RAG pipeline)
├── RAG_Using_LangChain.ipynb      # Notebook: step-by-step RAG development
└── README.md
```

## 🔮 Future Improvements

- Multi-video chat
- Conversation memory for follow-up questions
- Source timestamps with each answer
- Support for more LLM providers

## 👤 Author

**Mujahid Hussain**: Python Developer & Automation Specialist

⭐ If you found this useful, please star the repo!
