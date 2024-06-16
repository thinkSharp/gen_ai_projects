from youtube_transcript_api import YouTubeTranscriptApi
import re

def get_video_id(url):
    # Extract the video ID from the YouTube URL
    video_id = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
    if video_id:
        return video_id.group(1)
    else:
        raise ValueError("Invalid YouTube URL")

def fetch_transcript(video_id):
    try:
        # Fetch the transcript using the video ID
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([entry['text'] for entry in transcript_list])
        return transcript
    except Exception as e:
        return str(e)


def get_transcript(url):
    try:
        video_id = get_video_id(url)
        transcript = fetch_transcript(video_id)
    except ValueError as ve:
        print(ve)
    except Exception as e:
        print (f"An error occured: {e}")

    return transcript

"""
def main():
    url = input("Enter YouTube URL: ")
    try:
        video_id = get_video_id(url)
        transcript = fetch_transcript(video_id)
        print("Transcript of the video:")
        print(transcript)
    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
"""