# Video RAG

This system enables users to interact with video content (uploaded or from YouTube) using text or image queries. It leverages video processing, multimodal embedding (CLIP), and vector similarity search to retrieve relevant information from videos and respond naturally through a language model.

---

## Table of Contents

1. [System Workflow](#1-system-workflow)

    1.1. [Tech Stack](#11-tech-stack)  

    1.2. [System Explanation](#12-system-explanation)

2. [UI](#2-ui)

3. [Quick Run](#3-quick-run)

4. [Limitations](#4-limitations)

5. [Future Work](#5-future-work)

---

## 1. System Workflow 
![alt text](/assets/architecture.png)
### 1.1 Tech Stack

- **Backend**:
  - **Streamlit** – For building the web interface and user interaction.
  - **HuggingFace CLIP** – For encoding video content (image and text) into embeddings.
  - **Faiss** – For efficient similarity search on vector embeddings.
  - **Cohere** – For generating responses using a language model.
  - **Whisper** – For transcribing audio from video files.

- **Frontend**:
  - **Streamlit** – To provide an interactive user interface for video processing and querying.
  
- **Video Processing**:
  - **FFmpeg** – For video conversion and frame extraction.
  - **OpenCV** – For handling video frame extraction and image processing.

- **Dependencies**:
  - **Pillow** – For image processing.
  - **MoviePy** – For handling video editing and audio extraction.

- **Deployment**:
  - **Docker** – For containerizing the application for easy deployment.
  - **Miniconda** – For creating isolated Python environments.
  
### 1.2 System explanation
The **RAG** system follows a structured workflow for processing video content and allowing semantic search and interaction. Below is an overview of the main steps in the system workflow:

1. **User Input**:
   - The user can provide input through either a **YouTube link** or by **uploading a video file**.

2. **Video Processing**:
   - The uploaded video or YouTube video is stored, and the frames are extracted from the video. This helps in visualizing the content for further analysis.

3. **CLIP Model Encoding**:
   - The **CLIP model** is used to encode both image and text data. This is important for extracting the semantic meaning of the video’s visual content and captions.

4. **Storage of Vector Data**:
   - The video’s image and text vector data are stored for future reference in vector stores. These embeddings are used to enable similarity search and context retrieval.

5. **Retrieval and Response**:
   - When the user queries the system, it uses **Faiss** (a vector similarity search engine) to find the most relevant context from the stored vector data. The retrieved context is then used to generate a response through an **LLM agent**.

6. **Final Output**:
   - The system returns a response based on the context retrieved from the video. The response can provide relevant information from the video, such as summaries or answering specific questions about the video content.

## 2. UI
- The Streamlit UI 

![alt text](/assets/UI.png)
## 3. Quick Run

1. First, clone the repository to your local machine:
```bash
git clone https://github.com/DatTruonggg/eklipse_pro1.git
cd eklipse_pro1
```
2. Set Up a New Conda Environment

```bash
conda create -n eklipseLLM python=3.10
conda activate eklipseLLM
```

3. Install Dependencies

```bash
pip install -r requirements.txt
```
4. Run the Application. Once the application starts, you can open the URL provided by **Streamlit** (typically `http://localhost:8501`) in your web browser.
Start the **Streamlit** application:

```bash
make run
```

---

## 4. Limitations

While **EklipseLLM** provides powerful video content interaction and search capabilities, it has the following limitations:

1. **No Keyframe Storage for Image Search**:

   * The system does not currently support storing keyframes from the video for image-based queries. As a result, when performing an **image-to-text** query (searching for a timestamp based on a given frame), the system will not return a result unless keyframes are explicitly stored in the database.

2. **Image-to-Text and Text-to-Image Functionality Limitations**:

   * **Image-to-Text**: This feature allows users to upload an image from a frame and search for the metadata (timestamp) of that frame in the video. However, due to the lack of keyframe storage, the system cannot currently provide results for this search.
   * **Text-to-Image**: Users can query the system to find the frame corresponding to a specific text. However, this feature will only work if the system has a stored image for that text context, which it currently does not.

3. **Limited Data Storage for Keyframes**:

   * There is no dedicated **data storage** implemented for storing the keyframes and associated metadata. This means that currently, the system can only work with the images and text already processed and embedded, and users cannot add new images or keyframes after the initial processing.

4. **Code Refactoring and Optimizations**:

   * The current codebase is not fully refactored or optimized. This can impact both maintainability and scalability in the long run. Future improvements will focus on cleaning up and optimizing the code for better performance, flexibility, and readability.

5. **Hardware Requirements**:

   * The application requires significant computational resources for video processing, especially with **GPU acceleration**. Running on a **CPU-only environment** will result in slower performance, particularly with large videos.

---

## 5. Future Work

* **FastAPI**: To create a RESTful API for seamless integration with other services and to improve scalability.
* **Docker**: To containerize the application for easier deployment and consistency across environments.
* **GCS (Google Cloud Storage)**: For storing keyframe images and other assets in the cloud for scalable and reliable storage.
* **CI/CD (Jenkins)**: To automate testing, building, and deployment processes, improving the development workflow.
* **GKE (Google Kubernetes Engine)**: For managing the deployment, scaling, and orchestration of containerized applications.
* **Nginx**: To act as a reverse proxy and load balancer for handling traffic and ensuring high availability.
* **Helm Charts**: For managing Kubernetes applications and simplifying deployments in different environments.
* **Prometheus & Grafana**: For real-time monitoring of the system and visualizing performance metrics.

---
