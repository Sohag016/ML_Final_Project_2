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

## Project Overview
This project aims to perform sentiment analysis on Amazon product reviews to predict whether a review is positive or negative. The project involves the use of multiple machine learning algorithms for classification and their comparison based on various evaluation metrics.

The primary objective of this project is to analyze textual reviews and predict the sentiment of the reviews (Positive or Negative). Different machine learning models are trained on the dataset, and the best-performing model is selected based on accuracy and other evaluation metrics.

## Dataset Description
The dataset used in this project contains Amazon product reviews. The dataset has two main columns:
- **`reviewText`**: The textual review provided by customers.
- **`Positive`**: The sentiment label, where 1 indicates a positive sentiment, and 0 indicates a negative sentiment.

### Dataset Source
- The dataset is publicly available through various sources, such as Kaggle or GitHub repositories. You can use a similar dataset from [this link](https://raw.githubusercontent.com/rashakil-ds/Public-Datasets/refs/heads/main/amazon.csv).

## Data Preprocessing
Data preprocessing is a critical step to ensure the model can effectively learn from the data. The following steps were performed during preprocessing:

1. **Handling Missing Values**: 
   - Rows with missing values in the `reviewText` or `Positive` columns are removed to avoid issues during model training.

2. **Text Preprocessing**:
   - **Convert to Lowercase**: All text is converted to lowercase to standardize the text data.
   - **Remove Punctuation and Special Characters**: Non-alphanumeric characters are removed to focus on words that are meaningful for sentiment analysis.
   - **Stopwords Removal**: Commonly occurring words like "the", "is", "and", etc., are removed as they don’t provide much useful information.
   - **Tokenization**: The text is split into individual words (tokens).
   - **Lemmatization**: Words are reduced to their base or root form (e.g., "running" becomes "run").

3. **Data Splitting**:
   - The dataset is split into a training set and a testing set. Typically, 80% of the data is used for training and 20% for testing the model.

## Model Selection
Several machine learning models are selected to perform sentiment classification. The models chosen are:
1. **Logistic Regression**: A simple and efficient model for binary classification.
2. **Random Forest**: An ensemble method that works well for classification tasks.
3. **Support Vector Machine (SVM)**: A powerful classifier that works well for high-dimensional spaces.
4. **Naïve Bayes**: A probabilistic classifier based on Bayes’ Theorem, which is particularly effective for text classification tasks.
5. **XGBoost**: A gradient boosting method known for its high performance in classification tasks.

Each of these models has its strengths, and the goal is to compare them to find the best performer for sentiment analysis.

## Model Training
The models are trained using the **TF-IDF (Term Frequency-Inverse Document Frequency)** Vectorizer to transform the review text into numerical features. TF-IDF is widely used for text classification tasks as it reflects the importance of words in the context of the entire dataset.

Once the data is vectorized, each model is trained on the training set. The performance of the models is then evaluated on the test set.

## Formal Evaluation
The models are evaluated using various performance metrics to assess their ability to predict sentiments accurately. The following evaluation metrics are used:

1. **Accuracy**: Measures the proportion of correct predictions made by the model.
2. **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
3. **Recall**: The ratio of correctly predicted positive observations to all the observations in the actual class.
4. **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two.
5. **Confusion Matrix**: A table used to describe the performance of a classification model by showing the true positives, false positives, true negatives, and false negatives.

These metrics provide insight into how well the models are performing in terms of both classifying positive and negative reviews.

## Hyperparameter Tuning
To further improve the performance of the models, hyperparameter tuning is performed using techniques such as **Grid Search** and **Random Search**. This process involves searching for the best combination of parameters for each model to maximize performance. 

For example, in the case of Random Forest, parameters like `n_estimators` (number of trees) and `max_depth` (maximum depth of the trees) are tuned.

By optimizing these parameters, we can achieve better results for each model.

## Comparative Analysis
After evaluating all the models, a comparative analysis is done to understand the performance of each model in terms of accuracy, precision, recall, and F1 score. The results are visualized using a bar chart, which makes it easier to compare the models.

The model with the highest accuracy and best balance of precision, recall, and F1 score is considered the best model for sentiment analysis in this project.

## Conclusion & Comments
### Best Model:
Based on the evaluation metrics, **Logistic Regression** emerges as the best-performing model, with the following performance:
- **Accuracy**: 0.89
- **Precision**: 0.90
- **Recall**: 0.96
- **F1 Score**: 0.93

### Key Observations:
- **Text Preprocessing** plays a crucial role in improving model performance, as cleaning and standardizing the text helps the model focus on important features.
- **Hyperparameter Tuning** can significantly improve the performance of the models. The optimal settings for each model should be explored to obtain the best results.
- **Model Performance**: While Logistic Regression performed the best overall, other models like Random Forest and XGBoost also provided competitive results and may be preferable in different contexts or datasets.

### Future Work:
- Experiment with more advanced models like **Deep Learning** (LSTM, CNN).
- Enhance text preprocessing by handling sarcasm or using more advanced NLP techniques.
- Explore other sentiment analysis datasets to see how the models perform in different domains.

---



