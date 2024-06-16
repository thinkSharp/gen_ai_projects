import torch
import gradio as gr

# Use a pipeline as a high-level helper
from transformers import pipeline

model_path = '../Models/models--google-t5--t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4'

language_codes = {
    'English': 'en',
    'French': 'fr',
    'German': 'de',
    'Spanish': 'es',
    'Italian': 'it',
    'Dutch': 'nl',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Chinese': 'zh',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Arabic': 'ar'
}
def language_translator(source, target,text):
    task = f'translation_{language_codes[source]}_to_{language_codes[target]}'
    translator = pipeline(task, model=model_path)
    # translator = pipeline(task, model="google-t5/t5-small")
    translation = translator(text)

    return translation[0]['translation_text']

gr.close_all()


demo = gr.Interface(language_translator,
                    inputs=[gr.Dropdown(['English','French','German','Spanish','Italian','Dutch','Portuguese','Russian','Chinese','Japanese','Korean','Arabic'],
                                        label='Source Language', value='English' ),
                            gr.Dropdown(['English','French','German','Spanish','Italian','Dutch','Portuguese','Russian','Chinese','Japanese','Korean','Arabic'],
                                        label='Target Language', value='French' ),
                            gr.Textbox(label='Text to Translate', lines=5)],

                    outputs=[gr.Textbox(label="Translated Text", lines=5)],
                    title="Gen AI Learning Project 4: Language Translator",
                    description="This Application uses 'google-t5/t5-small' llm to provide translation service")
demo.launch()