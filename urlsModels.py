# import pandas as pd
# import re
# import string
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, accuracy_score
#
# # ====================
# # 1. Load Dataset
# # ====================
# df = pd.read_csv("data/merged_urls.csv")
#
# # Ensure the correct columns are present
# if 'url' not in df.columns or 'label' not in df.columns:
#     raise ValueError("CSV must contain 'url' and 'label' columns.")
#
# print("Dataset Loaded ✅")
# print(df.head())
#
# # ====================
# # 2. Text Preprocessing Function
# # ====================
# def preprocess_text(text):
#     if not isinstance(text, str):
#         text = str(text)
#     text = text.lower()  # lowercase
#     text = text.strip()  # remove leading/trailing spaces
#     return text
#
# df['clean_url'] = df['url'].apply(preprocess_text)
#
# # ====================
# # 3. Train-Test Split
# # ====================
# X = df['clean_url']
# y = df['label']
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
#
# # ====================
# # 4. TF-IDF Vectorization
# # ====================
# vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)
#
# # ====================
# # 5. Train Model
# # ====================
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train_tfidf, y_train)
#
# # ====================
# # 6. Evaluation
# # ====================
# y_pred = model.predict(X_test_tfidf)
#
# print("\nModel Evaluation ✅")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
#
# # ====================
# # 7. Save Model & Vectorizer
# # ====================
# joblib.dump(model, "url_classifier_model.pkl")
# joblib.dump(vectorizer, "url_tfidf_vectorizer.pkl")
#
# print("\nModel and vectorizer saved successfully ✅")
#
# # ====================
# # 8. Prediction Example
# # ====================
# def predict_url(url):
#     clean = preprocess_text(url)
#     vector = vectorizer.transform([clean])
#     prediction = model.predict(vector)[0]
#     return prediction
#
# # Example usage
# example_url = "https://example.com/free-gift"
# print(f"\nExample Prediction for '{example_url}':", predict_url(example_url))

#
# # url_model_training.py
#
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, accuracy_score
# from scipy.sparse import hstack, csr_matrix
# import joblib
#
# # Load dataset with extracted features
# df = pd.read_csv("data/urls_with_features.csv")
#
# # Columns to use as numeric features
# numeric_features = ['url_length', 'dot_count', 'hyphen_count', 'has_at',
#                     'has_https', 'digits_count', 'ip_in_url', 'suspicious_word_count']
#
# # Prepare text and labels
# X_text = df['url']
# y = df['label']
#
# # Train-test split (stratify to keep distribution)
# X_text_train, X_text_test, y_train, y_test, X_num_train_df, X_num_test_df = train_test_split(
#     X_text, y, df[numeric_features], test_size=0.2, random_state=42, stratify=y
# )
#
# # TF-IDF Vectorization (text)
# vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
# X_train_tfidf = vectorizer.fit_transform(X_text_train)
# X_test_tfidf = vectorizer.transform(X_text_test)
#
# # Scale numeric features
# scaler = StandardScaler()
# X_train_num = scaler.fit_transform(X_num_train_df)
# X_test_num = scaler.transform(X_num_test_df)
#
# # Convert numeric features to sparse matrices
# X_train_num_sparse = csr_matrix(X_train_num)
# X_test_num_sparse = csr_matrix(X_test_num)
#
# # Combine TF-IDF and numeric features horizontally
# X_train_combined = hstack([X_train_tfidf, X_train_num_sparse])
# X_test_combined = hstack([X_test_tfidf, X_test_num_sparse])
#
# # Train Logistic Regression Model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train_combined, y_train)
#
# # Evaluate
# y_pred = model.predict(X_test_combined)
# print("Model Evaluation ✅")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
#
# # Save model and preprocessing objects
# joblib.dump(model, "url_classifier_model.pkl")
# joblib.dump(vectorizer, "url_tfidf_vectorizer.pkl")
# joblib.dump(scaler, "url_numeric_scaler.pkl")
#
# print("Model, vectorizer, and scaler saved successfully ✅")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack, csr_matrix
import joblib

# Load enhanced dataset
df = pd.read_csv("data/urls_with_enhanced_features.csv")

numeric_features = [
    'url_length', 'dot_count', 'hyphen_count', 'has_at', 'has_https',
    'digits_count', 'suspicious_char_count', 'subdomain_count', 'ip_in_url',
    'suspicious_word_count', 'url_entropy', 'is_shortener'
]

X_text = df['url']
y = df['label']

X_text_train, X_text_test, y_train, y_test, X_num_train_df, X_num_test_df = train_test_split(
    X_text, y, df[numeric_features], test_size=0.2, random_state=42, stratify=y
)

# Text vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=7000)  # added 3-grams & more features
X_train_tfidf = vectorizer.fit_transform(X_text_train)
X_test_tfidf = vectorizer.transform(X_text_test)

# Scale numeric features
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_num_train_df)
X_test_num = scaler.transform(X_num_test_df)

# Convert numeric features to sparse format
X_train_num_sparse = csr_matrix(X_train_num)
X_test_num_sparse = csr_matrix(X_test_num)

# Combine
X_train_combined = hstack([X_train_tfidf, X_train_num_sparse])
X_test_combined = hstack([X_test_tfidf, X_test_num_sparse])

# Train RandomForest
model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train_combined, y_train)

# Evaluate
y_pred = model.predict(X_test_combined)
print("Model Evaluation ✅")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save objects
joblib.dump(model, "url_classifier_model.pkl")
joblib.dump(vectorizer, "url_tfidf_vectorizer.pkl")
joblib.dump(scaler, "url_numeric_scaler.pkl")

print("Model, vectorizer, and scaler saved successfully ✅")
