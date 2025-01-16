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
This project aims to predict the sentiment of Amazon product reviews based on their textual content. The sentiment is binary: 1 for positive and 0 for negative. Various machine learning models are used for sentiment classification, and their performances are compared.

## Dataset Description
The dataset consists of Amazon product reviews with two columns:
- `reviewText`: The textual review provided by the customer.
- `Positive`: The sentiment label (1 for positive, 0 for negative).

The dataset is publicly available via the following link:
- [Amazon Product Review Dataset](https://raw.githubusercontent.com/rashakil-ds/Public-Datasets/refs/heads/main/amazon.csv)

## Data Preprocessing
The data preprocessing steps include:
- **Handling Missing Values**: Removed rows with missing `reviewText` or `Positive` columns.
- **Text Preprocessing**:
  - Convert all text to lowercase.
  - Remove stop words, punctuation, and special characters.
  - Tokenize and lemmatize the text data.
- **Data Splitting**: The dataset is split into training and testing sets.

### Code for Data Preprocessing:
```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Data Preprocessing Function
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatize and remove stopwords
    return ' '.join(tokens)

# Apply the preprocessing to the 'reviewText' column
df.dropna(subset=['reviewText', 'Positive'], inplace=True)
df['cleanedText'] = df['reviewText'].apply(preprocess_text)
## Model Selection
Several machine learning models are selected for sentiment classification:

Logistic Regression
Random Forest
Support Vector Machine (SVM)
Naïve Bayes
XGBoost
Code for Model Selection:
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": LinearSVC(random_state=42),
    "Naïve Bayes": MultinomialNB(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}
## Model Training
The models are trained using the TF-IDF Vectorization technique to convert text data into numerical features, followed by fitting each model on the training data.

Code for Model Training:
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['cleanedText'], df['Positive'], test_size=0.2, random_state=42)

# Apply TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train the models
trained_models = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)  # Train the model on the training set
    trained_models[name] = model
## Formal Evaluation
After training the models, their performance is evaluated using various metrics:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Code for Evaluation:
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Evaluation
results = {}
for name, model in trained_models.items():
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results[name] = {
        "accuracy": accuracy,
        "precision": report['1']['precision'],
        "recall": report['1']['recall'],
        "f1_score": report['1']['f1-score']
    }

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
## Hyperparameter Tuning
Hyperparameter tuning is performed using Grid Search to optimize selected models such as Random Forest and Logistic Regression.

Code for Hyperparameter Tuning:
from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning for Random Forest
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid_search_rf.fit(X_train_tfidf, y_train)

## Comparative Analysis
A comparative analysis is performed to compare the performance of all models based on accuracy, precision, recall, and F1 score. The results are visualized using a bar plot.

Code for Comparative Analysis:
import pandas as pd

# Convert results into DataFrame
results_df = pd.DataFrame(results).T
results_df.plot(kind='bar', figsize=(12, 6))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xlabel('Models')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
## Conclusion & Comments
Best Model: Logistic Regression achieved the highest accuracy of 0.89, with precision of 0.90, recall of 0.96, and F1 score of 0.93.
Challenges: The most significant challenges were handling noisy text data and tuning hyperparameters for optimal performance.
Key Lessons: Text preprocessing plays a critical role in improving model accuracy. Also, hyperparameter tuning can significantly boost performance.
Code for Best Model Evaluation:
best_model_name = results_df['accuracy'].idxmax()
best_model = trained_models[best_model_name]
y_pred_best = best_model.predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred_best)

# Display confusion matrix for best model
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Print Best Model Performance
print(f"Best Model: {best_model_name}")
print(f"Accuracy: {results[best_model_name]['accuracy']:.2f}")
print(f"Precision: {results[best_model_name]['precision']:.2f}")
print(f"Recall: {results[best_model_name]['recall']:.2f}")
print(f"F1 Score: {results[best_model_name]['f1_score']:.2f}")







