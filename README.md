# Customer Sentiment Analyzer

## Overview
Customer Sentiment Analyzer is a Machine Learning-based web application that predicts the sentiment of Amazon customer reviews as **Positive**, **Neutral**, or **Negative**. The project uses Natural Language Processing (NLP) techniques for text preprocessing and a trained machine learning model for sentiment classification.

## Features
- Predicts customer sentiment from review text.
- Classifies reviews into Positive, Neutral, and Negative.
- Interactive web interface built using Streamlit.
- Text preprocessing using NLTK.
- Fast predictions using a trained machine learning model.

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Joblib

## Project Structure

```
Customer_Sentiment_Analyzer/
│── Customer_Sentiment_Analyzer_Amazon_3class.ipynb
│── app.py
│── sentiment_model.joblib
│── tfidf_vectorizer.joblib
│── amazon_reviews.csv
```

## Dataset
The project uses an Amazon customer reviews dataset for training and testing the sentiment analysis model.

## Model
- Text Vectorization: TF-IDF Vectorizer
- Machine Learning Algorithm: Logistic Regression
- Sentiment Classes:
  - Positive
  - Neutral
  - Negative

## How to Run

1. Clone the repository.
2. Install the required Python libraries.
3. Run the Streamlit application:

```bash
streamlit run app.py
```

4. Open the local URL shown in the terminal.
5. Enter a customer review and click **Predict** to view the sentiment.

## Author

**Yalala Srinavya**

B.Tech Student

Machine Learning & Python Enthusiast
