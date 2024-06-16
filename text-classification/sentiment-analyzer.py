import torch
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

# Use a pipeline as a high-level helper
from transformers import pipeline

model_path = '../Models/models--distilbert--distilbert-base-uncased-finetuned-sst-2-english/snapshots/714eb0fa89d2f80546fda750413ed43d93601a13'

analyzer = pipeline("text-classification", model=model_path)
#analyzer = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

#print(analyzer(["this product is good", "This product was quite expensive"]))

def sentiment_analyzer(input):
    sentiment = analyzer(input)
    return sentiment[0]['label']

def plot_sentiment_charts(sentiments):
    sentiment_counts = pd.Series(sentiments).value_counts()

    # Create bar chart
    fig1, ax1 = plt.subplots()
    sentiment_counts.plot(kind='bar', ax=ax1)
    ax1.set_title('Sentiment Bar Chart')
    ax1.set_xlabel('Sentiment')
    ax1.set_ylabel('Frequency')

    # Create pie chart
    fig2, ax2 = plt.subplots()
    sentiment_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Sentiment Pie Chart')
    ax2.set_ylabel('')  # Hide the y-label for the pie chart

    return fig1, fig2
def analyze_reviews(file_path):
    reviews = []
    sentiments = []

    with open(file_path, 'r') as file:
        for line in file:
            review = line.strip()
            sentiment = sentiment_analyzer(review)
            reviews.append(review)
            sentiments.append(sentiment)

    data = {'Review': reviews, 'Sentiment': sentiments}
    df = pd.DataFrame(data)
    bar_chart, pie_chart = plot_sentiment_charts(sentiments)
    return df, bar_chart, pie_chart



# Example usage:
# sentiments = ['positive', 'negative', 'neutral', 'positive', 'negative']
# bar_chart, pie_chart = plot_sentiment_charts(sentiments)
# bar_chart.show()
# pie_chart.show()

#print(analyze_reviews("../training-data/user_reviews_sentiment.csv"))

gr.close_all()


demo = gr.Interface(analyze_reviews,
                    inputs=[gr.File(file_types=['csv'], label='Upload your review comment file')],

                    outputs=[gr.DataFrame(label="Reviews and Sentiments"), gr.Plot(label='Bar Chart'), gr.Plot(label='Pie Chart')],
                    title="Gen AI Learning Project 3: Review Sentiment analysis",
                    description="This Application will be used for analying sentiment of reviews")
demo.launch()