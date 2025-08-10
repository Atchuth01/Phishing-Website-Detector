


# Phishing Detection System

A comprehensive phishing detection web application built with **Streamlit** that detects phishing URLs and phishing websites from screenshots/images using machine learning models.

## Project Overview

Phishing attacks remain one of the most common cybersecurity threats, where attackers impersonate legitimate websites to steal sensitive information. This project provides a dual-method detection system:

1. **URL-based detection** using a hybrid machine learning model combining text features (TF-IDF) and engineered lexical URL features.

2. **Image-based detection** by analyzing website screenshots using a Convolutional Neural Network (CNN).

The system helps users identify potentially malicious URLs and websites through an intuitive web interface.

---

## Features

### URL Phishing Detection
- Uses TF-IDF vectorization on URL strings combined with numerical lexical features such as URL length, subdomain count, suspicious character counts, entropy, and more.
- Logistic Regression classifier trained on a large labeled URL dataset.
- Whitelist of popular safe domains to reduce false positives.

### Image Phishing Detection
- CNN model trained on website screenshots labeled as phishing or legitimate.
- Analyzes visual cues in screenshots to detect phishing attempts.

### User Interface
- Built with Streamlit for easy deployment and interactive user experience.
- Upload URL or image and get real-time phishing predictions with confidence scores.
- Clear warnings and safe confirmations shown based on model output.

---

## Installation

### Prerequisites

- Python 3.8 or higher  
- `pip` package manager  

### Steps

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/phishing-detection.git
    cd phishing-detection
    ```

2. **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate     # Windows
    ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Add trained model files:**  
   Download or place the following files in the project root directory:
   - `url_classifier_model.pkl`  
   - `url_tfidf_vectorizer.pkl`  
   - `url_numeric_scaler.pkl`  
   - `imgModel.h5`

---

## Usage

### Run the Streamlit application:

```bash
streamlit run app.py
````

This will open the app in your default web browser.

### How to Use

* **URL Check:**
  Enter a URL into the input box and click **Analyze URL**.
  The app predicts if the URL is phishing or safe, displaying the confidence score.
  Whitelisted domains will automatically be marked safe.

* **Image Check:**
  Upload a screenshot/image of a website (.png, .jpg, .jpeg) and click **Analyze Image**.
  The CNN model will analyze the image and predict if it shows a phishing website.

---

## Model Details

### URL Phishing Detection Model

* **Features:**

  * TF-IDF vectorization of URL text (1-2 grams)
  * Numeric lexical features like URL length, dots, hyphens, '@' presence, HTTPS, digits, suspicious character count, subdomain count, presence of IP, suspicious words, entropy, URL shortening detection.

* **Model:** Logistic Regression trained on \~850K labeled URLs (balanced dataset with phishing and legitimate samples).

* **Preprocessing:** Numeric features scaled using StandardScaler; TF-IDF features combined with numeric features.

### Image Phishing Detection Model

* **Architecture:** CNN with 3 Conv2D + MaxPooling layers, followed by Dense and Dropout layers.

* **Input:** Resized website screenshots (128x128 RGB).

* **Output:** Binary classification — phishing or safe.

---

## Dataset

* The URL dataset consists of \~858,740 samples labeled as phishing or legitimate, gathered from multiple public sources.

* Website screenshot images were collected and labeled for phishing and genuine sites for CNN training.

* Feature extraction scripts are included for URLs to generate numeric features.

---

## Training Pipeline

1. **Feature Extraction:**
   Extract lexical features from URLs including domain info, suspicious tokens, entropy, shortening services, and more.

2. **Text Vectorization:**
   Convert URL text into TF-IDF vectors (1-2 grams).

3. **Feature Scaling:**
   Apply StandardScaler to numeric features.

4. **Combine Features:**
   Merge TF-IDF sparse matrix with numeric feature matrix.

5. **Model Training:**
   Train Logistic Regression classifier on combined features.

6. **Evaluation:**
   Measure accuracy, precision, recall, and F1-score on a held-out test set.

7. **Image Model:**
   Train CNN on labeled website screenshots using TensorFlow Keras.

8. **Save Models:**
   Save all trained models and preprocessing objects for inference.

---

## Project Structure

```
phishing-detection/
│
├── streamlit_app.py           # Main Streamlit app script
├── url_classifier_model.pkl   # URL phishing classification model
├── url_tfidf_vectorizer.pkl   # TF-IDF vectorizer for URL text
├── url_numeric_scaler.pkl     # Scaler for URL numeric features
├── imgModel.h5                # CNN model for image phishing detection
├── requirements.txt           # Python dependencies
├── README.md                  # This documentation
├── data/                      # Raw and processed datasets (optional)
├── training_scripts/          # Scripts to preprocess and train models (optional)
└── utils/                     # Utility modules (optional)
```

---

## Future Improvements

* Incorporate more sophisticated lexical and semantic URL features (e.g., brand similarity, homoglyph detection).
* Use character-level embeddings or pretrained models (FastText, BERT) for URLs.
* Experiment with advanced classifiers such as Random Forest, XGBoost, or deep neural networks.
* Enhance image model with larger datasets and transfer learning from pre-trained CNNs.
* Add domain age and SSL certificate validation features.
* Implement real-time updating threat intelligence feeds for blacklists and whitelists.
* Build API for backend inference to integrate with other apps or browsers.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, suggestions, or collaboration, please open an issue or contact:

**Atchuth Vutukuri**
GitHub: https://github.com/Atchuth01

---

Thank you for checking out this phishing detection system! Stay safe online! 


