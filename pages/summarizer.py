import sys
import os
import subprocess
import tempfile
import numpy as np
import speech_recognition as sr
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer

# Add local bin directory to PATH
os.environ["PATH"] += os.pathsep + "/home/appuser/.local/bin"

# Install necessary packages if not available
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = ['transformers', 'sentencepiece', 'SpeechRecognition', 'youtube-transcript-api', 'yt-dlp']

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

# Specify the directory to save the model
MODEL_DIR = "models/bart-base"

# Load the BART model and tokenizer with cache directory
def load_model():
    print("Loading BART Base model...")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', cache_dir=MODEL_DIR)
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base', cache_dir=MODEL_DIR)
    return pipeline('summarization', model=model, tokenizer=tokenizer)

def download_video(video_id):
    """Download video using yt-dlp and extract audio."""
    print("Downloading video...")
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    audio_file_path = tempfile.mktemp(suffix='.wav')

    # Download audio using yt-dlp
    result = subprocess.run(['yt-dlp', '--extract-audio', '--audio-format', 'wav', '-o', audio_file_path, video_url, '--verbose'], capture_output=True)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to download audio: {result.stderr.decode()}")

    return audio_file_path

def transcribe_audio(audio_file_path):
    """Transcribe audio to text using SpeechRecognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)  # Read the entire audio file
        return recognizer.recognize_google(audio_data)

def main(desired_length):
    try:
        # Read the YouTube video ID from file
        with open('texts/video_id.txt', 'r') as file:
            youtube_video = file.read().strip()

        # Validate the video ID format
        video_id = youtube_video.split("=")[-1]
        if len(video_id) != 11:
            raise ValueError("Invalid YouTube video ID. Please provide a complete YouTube video ID.")

        print(f"Video ID: {video_id}")

        # Attempt to fetch the transcript for the video
        try:
            print("Fetching transcript...")
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            result = " ".join([t['text'] for t in transcript])
        except Exception as e:
            print("Transcript not available, attempting to download audio and transcribe...")
            audio_file_path = download_video(video_id)
            try:
                result = transcribe_audio(audio_file_path)
            finally:
                os.remove(audio_file_path)  # Clean up the audio file after processing

        # Calculate and print the number of words in the transcript
        num_words_in_transcript = len(result.split())
        print(f"Transcript Length: {len(result)} characters.")
        print(f"Number of words in transcript: {num_words_in_transcript} words.")

        # Initialize the summarizer pipeline with the loaded model
        summarizer = load_model()

        # Calculate how many iterations are needed for large texts
        chunk_size = 800  # Set the chunk size
        num_iters = len(result) // chunk_size + 1
        summarized_text = []

        # Summarize in chunks
        for i in range(num_iters):
            start, end = i * chunk_size, (i + 1) * chunk_size
            chunk_text = result[start:end]

            # Ensure the chunk is not empty
            if not chunk_text.strip():
                continue

            # Calculate min_length and max_length based on the desired length
            min_length = max(1, desired_length // num_iters)  # Ensure min_length is at least 1
            max_length = desired_length // num_iters  # Set max_length per chunk

            # Summarize the chunk
            out = summarizer(chunk_text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

            summarized_text.append(out)
            print(f"Chunk {i + 1}/{num_iters} summarized with length: {len(out.split())} words.")

        # Combine the summarized text
        final_summary = " ".join(summarized_text)
        final_summary_words = final_summary.split()

        # Trim the final summary to match the desired length
        if len(final_summary_words) > desired_length:
            final_summary = " ".join(final_summary_words[:desired_length])
        final_summary_length = len(final_summary.split())

        # Format the summary in bullet points
        bullet_points = final_summary.split('. ')
        formatted_summary = "\n".join([f"- {point.strip()}" for point in bullet_points if point])

        # Add summary length information as the first line
        formatted_summary = (
            f"Actual Summary Length: {final_summary_length} words.\n"
            f"Desired Summary Length: {desired_length} words.\n\n{formatted_summary}"
        )

        # Save the formatted summary to a file
        with open('texts/summary.txt', 'w') as file:
            file.write(formatted_summary)
        print("Summary saved to 'texts/summary.txt'.")

    except Exception as e:
        print(f"Error occurred: {e}", file=sys.stderr)
        print("Debugging information:")
        if 'video_id' in locals():
            print(f"Video ID: {video_id}")
        if 'result' in locals():
            print(f"Transcript Length: {len(result)}")
            print(f"Number of words in transcript: {num_words_in_transcript}")
        print(f"Desired Length: {desired_length}")
        sys.exit(1)

if __name__ == "__main__":
    # Get the desired summary length from command-line arguments
    desired_length = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(desired_length)
