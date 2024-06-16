import torch
import gradio as gr
from Utils import youtube_transcript

# use a transformer pipeline as high-level helper
from transformers import pipeline


model_path = "../Models/models--sshleifer--distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff"
model_text_summary = pipeline("summarization", model=model_path, torch_dtype=torch.bfloat16)

#model_text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", torch_dtype=torch.bfloat16)

def split_text(text, max_length):
    # Split text into chunks of max_length
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]


def summarize_text(text, chunk_max_length=1024, summary_ratio=0.5):
    # Split the text into chunks
    chunks = split_text(text, chunk_max_length)

    # Summarize each chunk and combine the results
    summaries = []
    for chunk in chunks:
        # Calculate max_length for summary based on the length of the chunk and summary_ratio
        chunk_length = len(chunk.split())
        summary_max_length = min(int(chunk_length * summary_ratio), chunk_max_length)
        # Ensure the summary max_length is less than the chunk length
        summary_max_length = max(30, summary_max_length)
        summary = model_text_summary(chunk, max_length=summary_max_length, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    # Combine all summaries into a final summary
    final_summary = ' '.join(summaries)
    return final_summary
def youtube_text_summary(url):
    transcript = youtube_transcript.get_transcript(url)
    print(transcript)
    print(len(transcript))
    output = summarize_text(transcript)
    print(output)
    return transcript, output




gr.close_all()


demo = gr.Interface(youtube_text_summary,
                    inputs=[gr.Text(label="Input Youtube URL", lines=1)],

                    outputs=[gr.Textbox(label="Youtube Transcript", lines=10), gr.Textbox(label="Summarized text",lines=4)],
                    title="Gen AI Learning Project 2: YouTube Text summarization using sshleifer/distilbart-cnn-12-6 model",
                    description="This Application will be used to summarize the Youtube URL")
demo.launch()
