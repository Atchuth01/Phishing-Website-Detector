# # streamlit_app.py
# import streamlit as st
# import joblib
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import re
# from scipy.sparse import hstack, csr_matrix
#
# # =======================
# # Load models & preprocessing objects
# # =======================
# @st.cache_resource
# def load_url_models():
#     vectorizer = joblib.load("url_tfidf_vectorizer.pkl")
#     scaler = joblib.load("url_numeric_scaler.pkl")
#     model = joblib.load("url_classifier_model.pkl")
#     return vectorizer, scaler, model
#
# @st.cache_resource
# def load_img_model():
#     return tf.keras.models.load_model("imgModel.h5")
#
# # Preprocess URL (match training: lowercase + strip only)
# def preprocess_url(url):
#     return url.lower().strip()
#
# # Extract numeric features same as training
# def extract_url_features(url):
#     url = url.lower().strip()
#     length = len(url)
#     dot_count = url.count('.')
#     hyphen_count = url.count('-')
#     has_at = 1 if '@' in url else 0
#     has_https = 1 if url.startswith('https') else 0
#     digits_count = sum(c.isdigit() for c in url)
#     ip_in_url = 0
#     try:
#         domain = re.findall(r'://([^/]+)/?', url)
#         domain = domain[0] if domain else url
#         import ipaddress
#         ipaddress.ip_address(domain)
#         ip_in_url = 1
#     except:
#         ip_in_url = 0
#     suspicious_words = ['free', 'login', 'secure', 'update', 'verify', 'account', 'bank', 'paypal', 'ebay', 'click']
#     suspicious_count = sum(word in url for word in suspicious_words)
#     return [length, dot_count, hyphen_count, has_at, has_https, digits_count, ip_in_url, suspicious_count]
#
# # Preprocess Image
# def preprocess_image(image, target_size=(128, 128)):
#     img = image.resize(target_size)
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array
#
# # Whitelist of popular safe domains
# SAFE_DOMAINS = [
#     "youtube.com", "google.com", "facebook.com", "twitter.com", "amazon.com",
#     "wikipedia.org", "linkedin.com", "apple.com", "microsoft.com", "instagram.com",
#     "reddit.com", "github.com", "stackoverflow.com", "netflix.com", "paypal.com",
#     "dropbox.com", "quora.com", "bbc.com", "cnn.com", "nytimes.com", "zoom.us",
#     "salesforce.com", "adobe.com", "spotify.com", "ebay.com", "tumblr.com",
#     "pinterest.com", "etsy.com", "discord.com", "slack.com", "trello.com",
#     "medium.com", "wordpress.com"
# ]
#
# def is_safe_domain(url):
#     url = url.lower()
#     for domain in SAFE_DOMAINS:
#         if domain in url:
#             return True
#     return False
#
# # =======================
# # Streamlit UI
# # =======================
# st.set_page_config(page_title="Phishing Detection", page_icon="🔍", layout="centered")
# st.title("🔍 Phishing Detection System")
# st.write("Detect phishing from URLs or website screenshots/images.")
#
# mode = st.radio("Choose Detection Type:", ["URL Check", "Image Check"])
#
# if mode == "URL Check":
#     st.subheader("🔗 URL Phishing Detection")
#     url_input = st.text_input("Enter URL to check:")
#     if st.button("Analyze URL"):
#         if url_input.strip() == "":
#             st.warning("Please enter a URL.")
#         elif is_safe_domain(url_input):
#             st.success("✅ Safe URL (Whitelisted).")
#         else:
#             vectorizer, scaler, url_model = load_url_models()
#             processed = preprocess_url(url_input)
#
#             # TF-IDF vector
#             tfidf_vec = vectorizer.transform([processed])
#
#             # Numeric features scaled
#             numeric_feats = extract_url_features(url_input)
#             numeric_scaled = scaler.transform([numeric_feats])
#
#             # Combine features
#             combined_features = hstack([tfidf_vec, csr_matrix(numeric_scaled)])
#
#             # Predict probability
#             proba = url_model.predict_proba(combined_features)[0][1]
#             threshold = 0.5
#             if proba > threshold:
#                 st.error(f"🚨 Phishing URL Detected! (Confidence: {proba:.2f})")
#             else:
#                 st.success(f"✅ Safe URL. (Confidence: {1 - proba:.2f})")
#
# elif mode == "Image Check":
#     st.subheader("🖼️ Image Phishing Detection")
#     uploaded_image = st.file_uploader("Upload a screenshot/image of the website", type=["png", "jpg", "jpeg"])
#     if uploaded_image is not None:
#         image = Image.open(uploaded_image).convert("RGB")
#         st.image(image, caption="Uploaded Image", use_column_width=True)
#
#         if st.button("Analyze Image"):
#             img_model = load_img_model()
#             processed_img = preprocess_image(image)
#             prediction = img_model.predict(processed_img)
#             if prediction[0][0] > 0.5:
#                 st.error("🚨 Phishing Website Detected from Image!")
#             else:
#                 st.success("✅ Website Appears Safe from Image.")


# streamlit_app.py
import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
from PIL import Image
import re
import ipaddress
from urllib.parse import urlparse
from scipy.sparse import hstack, csr_matrix
import math

# =======================
# Load models & preprocessing objects
# =======================
@st.cache_resource
def load_url_models():
    vectorizer = joblib.load("url_tfidf_vectorizer.pkl")
    scaler = joblib.load("url_numeric_scaler.pkl")
    model = joblib.load("url_classifier_model.pkl")
    return vectorizer, scaler, model

@st.cache_resource
def load_img_model():
    return tf.keras.models.load_model("imgModel.h5")

# Preprocess URL (match training: lowercase + strip only)
def preprocess_url(url):
    return url.lower().strip()

# Enhanced URL numeric feature extraction (matches training features)
SHORTENERS = [
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co", "bit.do", "shorte.st",
    "adf.ly", "is.gd", "cutt.ly", "buff.ly"
]

def url_entropy(s):
    prob = [float(s.count(c)) / len(s) for c in set(s)]
    entropy = -sum([p * math.log2(p) for p in prob])
    return entropy

def extract_url_features(url):
    url = url.lower().strip()
    length = len(url)
    dot_count = url.count('.')
    hyphen_count = url.count('-')
    has_at = 1 if '@' in url else 0
    has_https = 1 if url.startswith('https') else 0
    digits_count = sum(c.isdigit() for c in url)
    suspicious_chars = ['?', '=', '%', '#', '&']
    suspicious_char_count = sum(url.count(ch) for ch in suspicious_chars)

    domain = urlparse(url).netloc
    subdomain_count = domain.count('.') - 1 if domain else 0
    if subdomain_count < 0:
        subdomain_count = 0

    ip_in_url = 0
    try:
        ipaddress.ip_address(domain)
        ip_in_url = 1
    except:
        ip_in_url = 0

    suspicious_words = ['free', 'login', 'secure', 'update', 'verify', 'account', 'bank', 'paypal', 'ebay', 'click']
    suspicious_word_count = sum(word in url for word in suspicious_words)
    entropy = url_entropy(url)
    is_shortener = 1 if any(short in domain for short in SHORTENERS) else 0

    return [
        length, dot_count, hyphen_count, has_at, has_https,
        digits_count, suspicious_char_count, subdomain_count, ip_in_url,
        suspicious_word_count, entropy, is_shortener
    ]

# Preprocess Image
def preprocess_image(image, target_size=(128, 128)):
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Whitelist of popular safe domains (your original list)
SAFE_DOMAINS = [
    "youtube.com", "google.com", "facebook.com", "twitter.com", "amazon.com",
    "wikipedia.org", "linkedin.com", "apple.com", "microsoft.com", "instagram.com",
    "reddit.com", "github.com", "stackoverflow.com", "netflix.com", "paypal.com",
    "dropbox.com", "quora.com", "bbc.com", "cnn.com", "nytimes.com", "zoom.us",
    "salesforce.com", "adobe.com", "spotify.com", "ebay.com", "tumblr.com",
    "pinterest.com", "etsy.com", "discord.com", "slack.com", "trello.com",
    "medium.com", "wordpress.com"
]

def is_safe_domain(url):
    url = url.lower()
    for domain in SAFE_DOMAINS:
        if domain in url:
            return True
    return False

# =======================
# Streamlit UI
# =======================
st.set_page_config(page_title="Phishing Detection", page_icon="🔍", layout="centered")
st.title("🔍 Phishing Detection System")
st.write("Detect phishing from URLs or website screenshots/images.")

mode = st.radio("Choose Detection Type:", ["URL Check", "Image Check"])

if mode == "URL Check":
    st.subheader("🔗 URL Phishing Detection")
    url_input = st.text_input("Enter URL to check:")
    if st.button("Analyze URL"):
        if url_input.strip() == "":
            st.warning("Please enter a URL.")
        elif is_safe_domain(url_input):
            st.success("✅ Safe URL (Whitelisted).")
        else:
            vectorizer, scaler, url_model = load_url_models()
            processed = preprocess_url(url_input)

            # TF-IDF vector
            tfidf_vec = vectorizer.transform([processed])

            # Numeric features scaled
            numeric_feats = extract_url_features(url_input)
            numeric_scaled = scaler.transform([numeric_feats])

            # Combine features
            combined_features = hstack([tfidf_vec, csr_matrix(numeric_scaled)])

            # Predict probability
            proba = url_model.predict_proba(combined_features)[0][1]
            threshold = 0.5
            if proba > threshold:
                st.error(f"🚨 Phishing URL Detected! (Confidence: {proba:.2f})")
            else:
                st.success(f"✅ Safe URL. (Confidence: {1 - proba:.2f})")

elif mode == "Image Check":
    st.subheader("🖼️ Image Phishing Detection")
    uploaded_image = st.file_uploader("Upload a screenshot/image of the website", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Image"):
            img_model = load_img_model()
            processed_img = preprocess_image(image)
            prediction = img_model.predict(processed_img)
            if prediction[0][0] > 0.5:
                st.error("🚨 Phishing Website Detected from Image!")
            else:
                st.success("✅ Website Appears Safe from Image.")
