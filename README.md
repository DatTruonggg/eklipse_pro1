---

# 📦 Video Content Understanding System – Architecture Overview

This system enables users to interact with video content (uploaded or from YouTube) using text or image queries. It leverages video processing, multimodal embedding (CLIP), and vector similarity search to retrieve relevant information from videos and respond naturally through a language model.

---

## 🧭 System Workflow
![alt text](/assets/architecture.png)

### 🧑‍💻 1. **User Input**
Users can:
- Upload a video file or input a YouTube link
- Ask questions via:
  - **Text prompt** (e.g., “When does the man walk the dog?”)
  - **Image input** (e.g., frame of interest)

---

## 📹 Video Processing Pipeline

This stage preprocesses the video and builds the searchable vector store.

### ➤ Step 1: **Store New Video**
- The uploaded or downloaded video is saved locally.
- YouTube videos are cached to prevent redundant downloads using video ID.

### ➤ Step 2: **Extract Keyframes**
- TransNetV2 or interval-based method is used to sample keyframes.
- Frames are saved to disk with associated timestamps.

### ➤ Step 3: **Encode Keyframes and Transcript**
- CLIP model is used to generate embeddings:
  - **Image embeddings**: from keyframes
  - **Text embeddings**: from transcript (via Whisper or YouTube captions)

### ➤ Step 4: **Store Vectors**
- FAISS is used to store:
  - `image vector data` from keyframes
  - `text vector data` from transcript
- Stored embeddings are used for similarity search during querying.

---

## 🔍 Query Phase (Search)

### 👁️‍🗨️ Input:
- User inputs **text** or **image**
- Both types are encoded using CLIP

### 🔄 Encode:
- Query is transformed into a vector (text/image)

### 🔎 Retriever (FAISS):
- Retrieves `top-k` most similar image/text segments based on cosine similarity
- Returns context metadata (e.g., timestamp, text)

---

## 🧠 LLM Agent (Chatbot)
- Uses retrieved context to generate a natural response
- Powered by an LLM (e.g., Cohere, OpenAI)

---

## 🖥️ Interface & Deployment

### 🧪 Streamlit
- User interface for uploading videos, entering queries, and viewing results.

### ⚡ FastAPI
- Provides API endpoints (future-proofing for multi-client support)

### 🐳 Docker
- Used for containerized deployment of the entire system.

---

## 🧩 Technologies Used

| Component        | Technology                    |
|------------------|-------------------------------|
| Video Download   | `yt_dlp`, `tempfile`          |
| Frame Extraction | `TransNetV2`, `OpenCV`        |
| Text Transcript  | `Whisper`, `YouTube API`      |
| Embedding Model  | `OpenAI CLIP`, `HF Embeddings`|
| Vector Storage   | `FAISS`                       |
| Backend Server   | `FastAPI`                     |
| UI Interface     | `Streamlit`                   |
| Containerization | `Docker`                      |

---

## 🧠 Smart Caching

- YouTube videos are cached by video ID (`/data/cache_videos/`)
- Avoids re-downloading and re-processing duplicate content

---

## ✅ Output
- Accurate **textual answers** grounded in video content
- **Timestamped results** for visual search via frame matching

---

> This architecture enables scalable, multimodal video search and Q&A—blending visual understanding with conversational AI.

---


![alt text](/assets/UI.png)

![alt text](/assets/architecture.png)