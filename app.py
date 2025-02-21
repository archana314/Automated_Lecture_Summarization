import streamlit as st
import ollama  # ✅ Import Ollama for Mistral-based refinement
from audio_extractor import download_audio, transcribe_audio_with_whisper
from youtube_transcript_api import YouTubeTranscriptApi  
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import os
import re
import tempfile
from moviepy.editor import VideoFileClip
import subprocess
import torch

# Fix OpenMP Conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["USE_MKL"] = "0"
os.environ["MKL_THREADING_LAYER"] = "GNU"

# Define the cache directory
CACHE_DIR = "./models"
MODEL_NAME = "google/long-t5-tglobal-base"

# Load the model and tokenizer from the cache
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

# Initialize the summarization pipeline with the loaded model and tokenizer
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

def extract_transcript_details(youtube_video_url):
    """Extracts transcript from YouTube video."""
    try:
        video_id = youtube_video_url.split("v=")[-1].split("&")[0]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([i['text'] for i in transcript_text])
        return transcript
    except Exception:
        return None  # Return None if transcript extraction fails

def extract_audio_from_video(video_file):
    """Extracts audio using ffmpeg for better speed."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        temp_video_path = temp_video.name

    audio_path = temp_video_path.replace(".mp4", ".wav")

    # Use ffmpeg to extract audio at a lower sample rate (16kHz)
    command = f'ffmpeg -i "{temp_video_path}" -ac 1 -ar 16000 "{audio_path}" -y'
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return audio_path

def generate_initial_summary(transcript_text):
    """Generates a summary efficiently using Long-T5 with FP16 acceleration."""
    MAX_INPUT_TOKENS = 4096
    MAX_OUTPUT_LENGTH = min(500, int(len(transcript_text.split()) * 0.3))
    
    if not transcript_text.strip():
        st.error("❌ Transcript is empty. Cannot generate summary.")
        return None

    words = transcript_text.split()
    if len(words) < 50:
        st.error("❌ Transcript is too short for summarization.")
        return None

    chunks = [" ".join(words[i:i + MAX_INPUT_TOKENS]) for i in range(0, len(words), MAX_INPUT_TOKENS)]
    summary_parts = []

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
        summarizer.model.to(device)

        for chunk in chunks:
            formatted_input = (
                "Summarize this transcript concisely. Ensure clarity, remove redundancy, "
                "and preserve key details:\n\n" + chunk
            )

            with torch.no_grad():  # Disables gradient calculations for faster inference
                summary = summarizer(formatted_input, max_length=MAX_OUTPUT_LENGTH, min_length=100, do_sample=False)

            if summary and 'summary_text' in summary[0]:
                summary_parts.append(summary[0]['summary_text'])
            else:
                st.warning("⚠️ Some chunks could not be summarized.")

        return " ".join(summary_parts)
    except Exception as e:
        st.error(f"❌ Summarization failed: {str(e)}")
        return None

def refine_summary_with_ollama(summary_text):
    """Refines the initial summary using Mistral via Ollama."""
    if not summary_text or len(summary_text.strip()) == 0:
        st.error("❌ Initial summary is empty. Cannot refine further.")
        return None

    try:
        response = ollama.chat(
            model="mistral",
            messages=[
                {"role": "system", "content": "Make this summary clear, concise, and well-structured."},
                {"role": "user", "content": f"Refine this summary:\n\n{summary_text}"}
            ]
        )
        return response['message']['content'].strip()
    except Exception as e:
        st.error(f"❌ Ollama refinement failed: {str(e)}")
        return None

def clean_summary(text):
    """Cleans and formats the summary for better readability."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text.capitalize()

# Streamlit UI
st.title("YouTube Video Summarizer (Multi-Stage Refinement)")

option = st.radio("Choose Input Type:", ["YouTube URL", "Upload Video"])

youtube_link = ""
uploaded_video = None

if option == "YouTube URL":
    youtube_link = st.text_input("Enter the YouTube Video URL:")
    if youtube_link:
        video_id = youtube_link.split("v=")[-1].split("&")[0]
        st.image(f"https://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

elif option == "Upload Video":
    uploaded_video = st.file_uploader("Upload your video file", type=["mp4", "mov", "avi"])

if st.button("Get Summary"):
    transcript_text = ""
    
    if option == "YouTube URL" and youtube_link:
        with st.spinner("Extracting transcript..."):
            transcript_text = extract_transcript_details(youtube_link)
    elif option == "Upload Video" and uploaded_video:
        with st.spinner("Extracting audio..."):
            audio_path = extract_audio_from_video(uploaded_video)
            if audio_path and os.path.exists(audio_path):
                with st.spinner("Transcribing audio..."):
                    transcript_text = transcribe_audio_with_whisper(audio_path)
    
    if transcript_text:
        with st.spinner("Generating initial summary..."):
            initial_summary = generate_initial_summary(transcript_text)
        if initial_summary:
            st.markdown("## Initial Summary:")
            st.write(initial_summary)
            with st.spinner("Refining summary with Mistral..."):
                refined_summary = refine_summary_with_ollama(initial_summary)
            if refined_summary:
                refined_summary = clean_summary(refined_summary)
                st.markdown("## Refined Summary:")
                st.write(refined_summary)
                st.download_button("📥 Download Summary", refined_summary, "summary.txt")
            else:
                st.error("❌ Refinement failed.")
        else:
            st.error("❌ Summarization failed.")
    else:
        st.error("❌ Could not retrieve transcript or audio.")
