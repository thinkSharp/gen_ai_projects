import torch
import gradio as gr

# use a transformer pipeline as high-level helper
from transformers import pipeline


#model_path = "../Models/models--sshleifer--distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff"
#model_text_summary = pipeline("summarization", model=model_path, torch_dtype=torch.bfloat16)

model_text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", torch_dtype=torch.bfloat16)

text = """
"
Narendra Damodardas Modi (Gujarati: [ˈnəɾendɾə dɑmodəɾˈdɑs ˈmodiː] ⓘ; born 17 September 1950)[b] is an Indian politician who has served as the 14th Prime Minister of India since 26 May 2014. Modi was the chief minister of Gujarat from 2001 to 2014 and is the Member of Parliament (MP) for Varanasi. He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), a right wing Hindu nationalist paramilitary volunteer organisation. He is the longest-serving prime minister outside the Indian National Congress.

Modi was born and raised in Vadnagar in northeastern Gujarat, where he completed his secondary education. He was introduced to the RSS at the age of eight. At the age of 18, he was married to Jashodaben Modi, whom he abandoned soon after, only publicly acknowledging her four decades later when legally required to do so. Modi became a full-time worker for the RSS in Gujarat in 1971. The RSS assigned him to the BJP in 1985 and he rose through the party hierarchy, becoming general secretary in 1998.[c] In 2001, Modi was appointed Chief Minister of Gujarat and elected to the legislative assembly soon after. His administration is considered complicit in the 2002 Gujarat riots,[d] and has been criticised for its management of the crisis. According to official records, a little over 1,000 people were killed, three-quarters of whom were Muslim; independent sources estimated 2,000 deaths, mostly Muslim.[11] A Special Investigation Team appointed by the Supreme Court of India in 2012 found no evidence to initiate prosecution proceedings against him.[e] While his policies as chief minister were credited for encouraging economic growth, his administration was criticised for failing to significantly improve health, poverty and education indices in the state.[f]

"""

#text_summary = model_text_summary(text)
#print(text_summary)

def text_summary(input):
    output = model_text_summary(input)
    return output[0]['summary_text']

gr.close_all()

#demo = gr.Interface(text_summary, inputs='text', outputs='text')

demo = gr.Interface(text_summary,
                    inputs=[gr.Textbox(label="Input text to summarize", lines=6)],
                    outputs=[gr.Textbox(label="Summarized text",lines=4)],
                    title="Gen AI Learning Project 1: Text summarization using sshleifer/distilbart-cnn-12-6 model",
                    description="This Application will be used to summarize the paste text")
demo.launch()