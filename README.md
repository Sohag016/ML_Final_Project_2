# Sentiment Analysis on Amazon Product Reviews

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Selection](#model-selection)
5. [Model Training](#model-training)
6. [Formal Evaluation](#formal-evaluation)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Comparative Analysis](#comparative-analysis)
9. [Conclusion & Comments](#conclusion--comments)

---

## Project Overview
This project aims to predict the sentiment of Amazon product reviews based on their textual content. The sentiment is binary: **1 for positive** and **0 for negative**. Various machine learning models are used for sentiment classification, and their performances are compared.

---

## Dataset Description
The dataset consists of Amazon product reviews with two columns:
- **`reviewText`**: The textual review provided by the customer.
- **`Positive`**: The sentiment label (**1 for positive, 0 for negative**).

The dataset is publicly available via the following link:  
[Amazon Product Review Dataset](https://raw.githubusercontent.com/rashakil-ds/Public-Datasets/refs/heads/main/amazon.csv)

---

## Data Preprocessing
The data preprocessing steps include:

1. **Handling Missing Values**: Removed rows with missing `reviewText` or `Positive` columns.
2. **Text Preprocessing**:
   - Convert all text to lowercase.
   - Remove stop words, punctuation, and special characters.
   - Tokenize and lemmatize the text data.
3. **Data Splitting**: The dataset is split into training and testing sets.

---

## Model Selection
Several machine learning models are selected for sentiment classification:
1. **Logistic Regression**: A simple linear model for binary classification.
2. **Random Forest**: An ensemble learning method for better accuracy and robustness.
3. **Support Vector Machine (SVM)**: A powerful classifier for high-dimensional data.
4. **Naïve Bayes**: A probabilistic approach well-suited for text data.
5. **XGBoost**: A high-performance gradient boosting framework.

---

## Model Training
The models are trained using the **TF-IDF Vectorization** technique to convert text data into numerical features. Each model is then trained on the processed training data.

---

## Formal Evaluation
The models are evaluated using the following metrics:
1. **Accuracy**: Percentage of correct predictions.
2. **Precision**: Correctly predicted positive observations out of total predicted positives.
3. **Recall**: Correctly predicted positive observations out of all actual positives.
4. **F1 Score**: The harmonic mean of precision and recall.
5. **Confusion Matrix**: Provides insights into classification performance.

---

## Hyperparameter Tuning
Hyperparameter tuning is performed using **Grid Search** to optimize selected models, such as:
- **Random Forest**: Tuning `n_estimators` and `max_depth`.
- **Logistic Regression**: Tuning regularization parameters.

---

## Comparative Analysis
A comparative analysis is performed to evaluate all models based on accuracy, precision, recall, and F1 score. The results are visualized using bar plots to identify the best-performing model.

---

## Conclusion & Comments
- **Best Model**: Logistic Regression achieved the highest accuracy of **0.89**, with precision **0.90**, recall **0.96**, and F1 score **0.93**.
- **Challenges**: Handling noisy text data and tuning hyperparameters were the key challenges.
- **Lessons Learned**:
  - Text preprocessing significantly impacts model performance.
  - Hyperparameter tuning can substantially improve results.

### Future Work:
1. Experiment with deep learning models such as LSTM and CNN for better performance.
2. Handle sarcasm and complex sentiments using advanced NLP techniques.
3. Test on additional datasets for robustness across domains.
