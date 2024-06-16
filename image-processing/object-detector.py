import gradio as gr
from PIL import Image, ImageDraw, ImageFont

# Use a pipeline as a high-level helper
from transformers import pipeline

model_path = '../Models/models--facebook--detr-resnet-50/snapshots/1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b'
object_detector = pipeline("object-detection", model=model_path)
#object_detector = pipeline("object-detection", model="facebook/detr-resnet-50")


def draw_detections(image, detections):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for obj in detections:
        score = obj['score']
        label = obj['label']
        box = obj['box']
        xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']

        # Draw the bounding box
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="red", width=2)

        # Draw the label and score
        text = f"{label}: {score:.2f}"
        text_bbox = draw.textbbox((xmin, ymin), text, font=font)
        text_location = (xmin, ymin - (text_bbox[3] - text_bbox[1])) if ymin - (text_bbox[3] - text_bbox[1]) > 0 else (xmin, ymin + (text_bbox[3] - text_bbox[1]))
        draw.rectangle(((xmin, text_location[1]), (xmin + (text_bbox[2] - text_bbox[0]), text_location[1] + (text_bbox[3] - text_bbox[1]))), fill="red")
        draw.text((xmin, text_location[1]), text, fill="white", font=font)

    return image

def detect_object_and_draw_boundary(image_path):
    raw_image = Image.open(image_path)
    model_output = object_detector(raw_image)
    bounary_image = draw_detections(raw_image, model_output)
    return raw_image, bounary_image

gr.close_all()


demo = gr.Interface(detect_object_and_draw_boundary,
                    inputs=[gr.File(file_types=['jpg','png','gif'], label='Upload an image to detect objects in it.')],

                    outputs=[gr.Image(label='Input Image', type='pil'), gr.Image(label='Image with boundries of the detected objects', type='pil')],
                    title="Gen AI Learning Project 6: Object detection and drawing boundry.",
                    description="This Application uses 'facebook/detr-resnet-5' to detect objects in the uploaded image and draw boundry around them.")
demo.launch()