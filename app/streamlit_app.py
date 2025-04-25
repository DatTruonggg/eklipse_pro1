import os
import time
import streamlit as st
import torch
from transformers import CLIPModel, CLIPProcessor
from langchain_community.embeddings import HuggingFaceEmbeddings

from logs import log
from utils.processor.video_processor import VideoProcessor
from utils.transcription.service import TranscriptionService
from utils.vectorstore.text import TextVectorStore
from utils.vectorstore.image import ImageVectorStore
from utils.vectorstore.manager import VectorStoreManager
from utils.context_retriever import ContextRetriever
from utils.chat_engine import ChatbotEngine


@st.cache_data(ttl=3600)
def process_video(video_source):
    log.info("Processing video")
    start_time = time.time()

    # Init services
    processor = VideoProcessor(video_source)
    transcriber = TranscriptionService()

    # Embedder models
    text_embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", 
            device_map="cpu")
    
    clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=os.path.expanduser("~/.cache/huggingface/clip"))
    
    # VectorStores
    text_store = TextVectorStore(embedder=text_embedder)
    image_store = ImageVectorStore(embedder=text_embedder, processor=clip_processor, model=clip_model)
    vectorstore = VectorStoreManager(text_store=text_store, image_store=image_store)
    
    # Step 2: Extract frames
    frames = processor.extract_frames(method='interval', interval=5)
    log.info(frames)
    log.info('done')
    # Step 3: Transcribe
    transcript_data = transcriber.get_transcription(video_source)


    # Step 4: Build vectorstore
    text_vectorstore = vectorstore.store_text(transcript_data['segments'])
    image_vectorstore = vectorstore.store_image(frames['image'], frames['timestamp'])

    log.info(f"Done processing in {time.time() - start_time:.2f}s")
    return text_vectorstore, image_vectorstore, transcript_data


def main():
    st.set_page_config(page_title="Video Content Chatbot", layout="wide")
    st.title("Video Content Chatbot")

    # Input type
    input_type = st.sidebar.radio("Choose Input Method", ["YouTube Link", "Upload Video"])

    # Video input
    if input_type == "YouTube Link":
        video_source = st.sidebar.text_input("Paste YouTube URL")
        uploaded_file = None
    else:
        video_source = None
        uploaded_file = st.sidebar.file_uploader("Upload .mp4/.mov", type=['mp4', 'mov'])

    if video_source or uploaded_file:
        try:
            with st.spinner("Processing..."):
                text_vectorstore, image_vectorstore, transcript_data = process_video(video_source or uploaded_file)

            # Init retriever & chatbot
            retriever = ContextRetriever(
                text_vectorstore=text_vectorstore,
                image_vectorstore=image_vectorstore,
                k=10
            )
            chatbot = ChatbotEngine()

            # UI Layout
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Ask about the video 👇")
                query = st.text_input("What do you want to know?")
                if st.button("Get Answer"):
                    context = retriever.retrieve_context(query)
                    response = chatbot.generate_response(context, query)
                    st.markdown("### Chatbot says:")
                    st.write(response)

            with col2:
                if video_source and "youtube.com" in str(video_source):
                    video_id = video_source.split("v=")[-1]
                    st.markdown("#### Embedded YouTube")
                    st.markdown(f'<iframe width="480" height="280" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)

                st.markdown("#### Transcript Preview")
                st.markdown(f"<div style='height: 350px; overflow-y: scroll; background-color: #f9f9f9; padding: 10px'>{transcript_data['text']}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")
            log.error(f"[App Error] {e}")
    else:
        st.info("Please upload a video or paste a YouTube URL to get started.")


if __name__ == "__main__":
    main()
