# Sentiment Classification Using Embeddings

This project focuses on classifying the sentiment of tweets into **Positive**, **Negative**, or **Neutral** categories using Advanced Natural Language Processing (NLP) techniques and Machine Learning.

## 🚀 Project Overview
The core idea of this project is to use **Sentence Transformers** to convert raw tweet text into high-dimensional mathematical vectors (embeddings). These embeddings capture the semantic meaning of the text, which are then fed into powerful classifiers to predict the sentiment accurately.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** - `pandas`, `numpy` (Data Manipulation)
  - `scikit-learn` (ML Algorithms & Evaluation)
  - `xgboost` (Gradient Boosting Classifier)
  - `sentence-transformers` (BERT-based Embeddings)
  - `matplotlib`, `seaborn`, `wordcloud` (Data Visualization)
  - `nltk` (Text Preprocessing)
  - `kagglehub` (Dataset Management)

## 📊 Dataset
The project uses the **Twitter Tweets Sentiment Dataset** from Kaggle.
- **Data Source:** Automatically downloaded via `kagglehub`.
- **Content:** Raw tweets with their corresponding sentiment labels.

## ⚙️ Workflow
1. **Data Loading:** Fetching the dataset using Kaggle API.
2. **Preprocessing:** Cleaning text, removing noise, and handling stopwords using NLTK.
3. **Feature Extraction:** Generating embeddings using the `all-MiniLM-L6-v2` (or similar) Sentence Transformer model.
4. **Model Training:** Training **Logistic Regression** and **XGBoost** models on the extracted embeddings.
5. **Evaluation:** Analyzing performance using Accuracy scores and Confusion Matrices.

## 📈 Results
The models were evaluated based on their ability to distinguish between different sentiments. Visualizations like **Word Clouds** were used to identify frequent terms in each sentiment category.

## 💻 How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost sentence-transformers matplotlib seaborn nltk wordcloud kagglehub
