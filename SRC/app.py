import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = f'Feiiisal/cardiffnlp_twitter_roberta_base_sentiment_latest_Nov2023'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict_tweet(tweet):
    inputs = tokenizer(tweet, return_tensors="pt", padding="max_length", max_length=128)
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1)
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    return {sentiment_classes[i]: float(probs.squeeze()[i]) for i in range(len(sentiment_classes))}

iface = gr.Interface(
    fn=predict_tweet,
    inputs="text",
    outputs="label",
    title="Vaccine Sentiment Classifier",
    description="Enter a text about vaccines to determine if the sentiment is negative, neutral, or positive.",
    examples=[
        ["Vaccinations have been a game-changer in public health, significantly reducing the incidence of many dangerous diseases and saving countless lives."],  
        ["Vaccinations are a medical intervention that introduces a vaccine to stimulate an individual’s immune response against a particular disease."],  
        ["Vaccines are rushed to the market without proper testing and are pushed by corporations that value profits over the well-being of the public."] 
    ]
)

iface.launch(server_name="0.0.0.0", server_port=7860)