import gradio as gr
import torch
from PIL import Image
import scipy.io.wavfile as wavfile

# Use a pipeline as a high-level helper
from transformers import pipeline

model_path = '../Models/models--Salesforce--blip-image-captioning-large/snapshots/2227ac38c9f16105cb0412e7cab4759978a8fd90'
caption_image = pipeline("image-to-text", model=model_path)
#caption_image = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")


tts_model_path = ("../Models/models--kakao-enterprise--vits-ljs/snapshots"
                   "/3bcb8321394f671bd948ebf0d086d694dda95464")
narrator = pipeline("text-to-speech", model=tts_model_path)
#narrator = pipeline("text-to-speech", model="kakao-enterprise/vits-ljs")

def generate_audio(text):
    # Generate the narrated text
    narrated_text = narrator(text)

    # Save the audio to a WAV file
    wavfile.write("output.wav", rate=narrated_text["sampling_rate"],
                  data=narrated_text["audio"][0])

    # Return the path to the saved audio file
    return "output.wav"
def caption_the_image(image):
    semantics = caption_image(images=image)
    return generate_audio(semantics[0]['generated_text'])


gr.close_all()

demo = gr.Interface(fn=caption_the_image,
                    inputs=[gr.Image(label="Select Image",type="pil")],
                    outputs=[gr.Audio(label="Generated Audio")],
                    title="@GenAILearniverse Project 8: Image Captioning with Audio",
                    description="This application uses 'Salesforce/blip-image-captioning-large' and 'kakao-enterprise/vits-ljs' to create text from image and narrate the captioning")
demo.launch()