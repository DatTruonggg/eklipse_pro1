import os
import streamlit as st
import time
import torch
from src.process_video import VideoProcessor
from src.embedding import EmbeddingService
from src.transcript import TranscriptionService
from src.retriever import ContextRetriever
from src.chat import ChatbotEngine
from src.store import VectorStore
from logs import log

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

@st.cache_data(ttl=3600)  # Cache for 1 hour
def process_video(video_source):
    """Cached video processing function"""
    log.info("Starting video processing")
    start_time = time.time()
    
    # Initialize services
    processor = VideoProcessor(video_source)
    transcriber = TranscriptionService()
    embedder = EmbeddingService()
    vectorstore = VectorStore()
    
    try:
        # Download/process video
        video_path = (processor.download_youtube_video() 
                      if 'youtube.com' in str(video_source) 
                      else processor.process_uploaded_file(video_source))
        log.info(f"Video downloaded/processed: {video_path}")
        
        # Extract frames (limit to reduce processing time)
        frames = processor.extract_frames()
        
        # Get Transcript
        if video_source and 'youtube.com' in str(video_source):
            transcript_data = transcriber.get_youtube_transcript(video_source)
        else:
            transcript_data = transcriber.get_transcription(video_source)

        context_vectorstore, image_vectorstore = vectorstore.embed_and_store(transcript_data)
        
        log.info(f"Video processing completed in {time.time() - start_time:.2f} seconds")
        
        return context_vectorstore, image_vectorstore, transcript_data
    
    except Exception as e:
        log.error(f"Video processing error: {e}")
        raise

def main():
    st.set_page_config(
        page_title="Video RAG", 
        page_icon="🎥", 
        layout="wide"
    )
    
    # Title
    st.title("🎬 Video RAG")
    
    # Input method selection in main content area
    input_type = st.radio(
        "Choose Input Method", 
        ["YouTube Link", "Video File Upload"]
    )
    
    # Video Input
    if input_type == "YouTube Link":
        video_source = st.text_input("Enter YouTube Video URL below:")
        uploaded_file = None
    else:
        video_source = None
        uploaded_file = st.file_uploader(
            "Upload Video File", 
            type=['mp4', 'avi', 'mov']
        )
    
    # Process Video
    if video_source or uploaded_file:
        try:
            # Use cached processing
            with st.spinner('Processing video...'):
                context_vectorstore, image_vectorstore, transcript_data = process_video(video_source or uploaded_file)
            
            # Chatbot and Context Retrieval
            retriever = ContextRetriever(context_vectorstore, image_vectorstore)
            chatbot = ChatbotEngine()

            # Main Content Layout: Two columns for better structure
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(
                    "<h1 style='color: #EC5331;'>📽️ Video Details</h1>",
                    unsafe_allow_html=True,
                )
                
                # Embed YouTube video using the video ID
                if video_source:
                    video_id = video_source.split("=")[-1]
                    st.markdown(
                        f'<iframe width="490" height="315" src="https://www.youtube.com/embed/{video_id}?start=90&autoplay=1" frameborder="0" allowfullscreen></iframe>',
                        unsafe_allow_html=True,
                    )
                    
                st.markdown("### Video Transcript")
                transcript_text = transcript_data.get("text", "")
                st.markdown(
                    f"<div style='height: 400px; overflow-y: scroll;'>{transcript_text}</div>",
                    unsafe_allow_html=True,
                )
            
            with col2:
                st.header("Video Q&A")
                query = st.text_input("Ask a question about the video content")

                if st.button("Get Answer"):
                    # Retrieve and generate response
                    text_context = retriever.retrieve_context(query)
                    log.info(f"Retrieved context: {text_context}")
                    response = chatbot.generate_response(
                        context=text_context, 
                        query=query
                    )
                    
                    st.markdown("### Response")
                    st.write(response)
            
        except Exception as e:
            st.error(f"Processing Error: {e}")
            log.error(f"Streamlit app error: {e}")
    
    else:
        st.markdown("### 🚀 Welcome to Video RAG")

if __name__ == "__main__":
    main()
