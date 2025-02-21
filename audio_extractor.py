import yt_dlp
import os
import whisper
import torch  # ✅ Import torch for GPU checking
from pydub import AudioSegment
from faster_whisper import WhisperModel

def download_audio(youtube_url, output_folder="audio_files"):
    """Downloads audio from a YouTube URL using yt-dlp and saves it as an MP3 file."""
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Extract video ID
        video_id = youtube_url.split("v=")[1].split("&")[0]
        output_path = os.path.join(output_folder, f"{video_id}.mp3")

        # yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_folder, f"{video_id}.%(ext)s"),  # Save in correct format
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': False
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        # ✅ Handle double ".mp3.mp3" issue safely
        downloaded_files = os.listdir(output_folder)
        for file in downloaded_files:
            if file.startswith(video_id) and file.endswith(".mp3"):
                corrected_name = f"{video_id}.mp3"
                if file != corrected_name:
                    os.rename(os.path.join(output_folder, file), os.path.join(output_folder, corrected_name))
                output_path = os.path.join(output_folder, corrected_name)  # Update output_path
        
        # ✅ Convert to MP3 if necessary
        if not output_path.endswith(".mp3"):
            new_output_path = os.path.join(output_folder, f"{video_id}.mp3")
            audio = AudioSegment.from_file(output_path)
            audio.export(new_output_path, format="mp3")
            os.remove(output_path)  # Remove the original file
            output_path = new_output_path
        
        # ✅ Ensure file exists before returning
        if not os.path.exists(output_path):
            print(f"❌ ERROR: File not found at {output_path}")
            return None
        else:
            print(f"✅ SUCCESS: Audio saved at {output_path}")
            return output_path

    except Exception as e:
        print(f"❌ ERROR in download_audio(): {e}")
        return None




def transcribe_audio_with_whisper(audio_path):
    """Transcribes audio using Faster Whisper for speed optimization."""
    model = WhisperModel("small", compute_type="float32")  # Use float32 for CPU support
  # Use "tiny" or "small" for more speed
    segments, _ = model.transcribe(audio_path)

    transcript = " ".join(segment.text for segment in segments)
    return transcript

