import torch
import gradio as gr
import fitz # PyMuPDF

# Use a pipeline as a high-level helper
from transformers import pipeline

model_path = '../Models/models--deepset--roberta-base-squad2/snapshots/cbf50ba81465d4d8676b8bab348e31835147541b'
qna = pipeline("question-answering", model=model_path)
#qna = pipeline("question-answering", model="deepset/roberta-base-squad2")


def read_pdf(file_path):
    try:
        # Open the PDF file
        pdf_document = fitz.open(file_path)

        # Initialize a string to store the content
        content = ""

        # Iterate through each page
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)  # Load the page
            content += page.get_text()  # Extract text from the page

        # Close the PDF file
        pdf_document.close()

        return content
    except Exception as e:
        return str(e)

def answer_question(file_path, question):
    context = read_pdf(file_path)
    answer = qna(question= question, context=context)
    return answer['answer']

gr.close_all()


demo = gr.Interface(answer_question,
                    inputs=[gr.File(file_types=['pdf'], label='Upload a pdf file'), gr.Textbox(label='Please ask question?', lines=3)],

                    outputs=[gr.Textbox(label='Answer is:', lines=3)],
                    title="Gen AI Learning Project 5: Question and Answer with PDF file",
                    description="This Application uses 'deepset/roberta-base-squad2' to simple short answer from PDF content to questions asked.")
demo.launch()

