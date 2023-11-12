# SentimentalBERT

## Introduction

This project focuses on fine-tuning a pre-trained sentiment analysis model to classify Twitter content, specifically targeting vaccine-related tweets. The objective is to categorize tweets into three sentiments: negative, neutral, and positive, enhancing the understanding of social media dynamics.

## Setup

### Prerequisites
- Python
- Required Libraries:
  - `accelerate`
  - `transformers`
  - `torch`
  - `pandas`
  - `numpy`
  - `datasets`
  - `huggingface_hub`
  - `contractions`
    
### Installing Necessary Libraries
Install the required libraries using the following command:
```bash
pip install accelerate transformers torch pandas numpy datasets huggingface_hub contractions
```
## Login to Hugging Face Hub

Use the notebook_login() function to log in to the Hugging Face Hub for accessing and managing datasets and models.

## Author

**Feisal Hassan**

An enthusiast in machine learning and natural language processing

## Article Link

For a detailed walkthrough of the project, refer to the article: Insightful Analytics: [Fine-Tuning Pre-trained Models for Robust Twitter Sentiment Analysis](https://medium.com/@feisalhassan77/insightful-analytics-fine-tuning-ai-for-robust-twitter-sentiment-analysis-8b770ffd6edb)

## App Screenshots

![App Screenshot](./Images/App_Screenshot.png)


## Deployed App Link

Access the deployed sentiment analysis web application here: [Deployed App](https://huggingface.co/spaces/Feiiisal/Twitter_Sentiment_Analysis_App)

## Exploratory Data Analysis

The project includes an extensive exploratory data analysis (EDA) section, visualizing sentiment distributions, tweet lengths, and word clouds for different sentiments.

## Data Cleaning and Preprocessing

The project covers data cleaning and preprocessing steps, including custom tweet cleaning functions and handling missing values and duplicates.

## Model Fine-Tuning and Evaluation

Details the process of [fine-tuning the cardiffnlp/twitter-roberta-base-sentiment-latest model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), including class weights calculation, tokenizer loading, data tokenization, custom training class, and evaluation metrics visualization.

## Deployment

The sentiment analysis model is deployed as an interactive web application using Gradio, Docker, and hosted on Hugging Face Spaces.

## Conclusion

This project demonstrates the process of fine-tuning a sentiment analysis model for Twitter, providing insights into public opinion and aiding businesses, policymakers, and individuals in decision-making.

