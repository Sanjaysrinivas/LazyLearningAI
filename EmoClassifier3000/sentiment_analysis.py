# sentiment_analysis.py
# MoodSwingAnalyzer: Because we trust machines to understand our feelings better than our friends.

# Step 1: Import Libraries
# We're importing the necessary libraries. TextBlob will be our quick and dirty sentiment analyzer, 
# while BERT from HuggingFace is our heavyweight AI that dives deeper into the nuances of your mood.

from textblob import TextBlob
from transformers import pipeline

# Step 2: Define Sentiment Analysis using TextBlob
# Why? Because TextBlob is fast, simple, and pretends to understand what you're feeling with minimal effort.

def textblob_sentiment(text):
    # TextBlob is about to judge your emotions—get ready.
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity  # This gives us a score from -1 (negative) to 1 (positive).

    # Now, based on that score, we slap a label on your emotions.
    if sentiment_score > 0:
        return "Positive"  # Clearly, you're in a great mood... or faking it.
    elif sentiment_score < 0:
        return "Negative"  # You probably stubbed your toe or hate Mondays.
    else:
        return "Neutral"  # You're indifferent, possibly stuck in traffic.
    

# Step 3: Define Sentiment Analysis using BERT (HuggingFace Transformers)
# Why? Because if you really want to analyze emotions in detail, 
# you call in BERT—the know-it-all model that leaves no feelings unexplored.

def bert_sentiment(text):
    # BERT takes sentiment analysis very seriously, unlike TextBlob. It uses cutting-edge transformer models to "understand" emotions.

    sentiment_model = pipeline("sentiment-analysis", model= 'distilbert-base-uncased-finetuned-sst-2-english')  # We load the pre-trained sentiment analysis model from HuggingFace.
    result = sentiment_model(text)  # BERT analyzes the text and returns a result.
    
    # BERT doesn't just guess. It comes with a label (Positive/Negative) and a confidence score.
    return result[0]['label'], result[0]['score']  # We get the sentiment and BERT’s confidence in its answer.


# Step 4: Test our sentiment analysis functions
# Why? Because all this code is useless unless we try , it out on some sample text to see if our models understand emotions better than your average human.
if __name__ == "__main__":
    # Let's test our models with a sample text
    test_text = "The weather is nice"  # Clearly a happy sentence, or so we think.
    
    # TextBlob Analysis
    print("TextBlob Sentiment:")
    print(f"Sentiment: {textblob_sentiment(test_text)}")  # TextBlob's quick opinion on your happiness.

    # BERT Analysis
    print("\nBERT Sentiment:")
    label, confidence = bert_sentiment(test_text)  # Let's see what BERT thinks.
    print(f"Sentiment: {label}, Confidence: {confidence:.2f}")  # BERT, as always, provides its sentiment with statistical backup.
