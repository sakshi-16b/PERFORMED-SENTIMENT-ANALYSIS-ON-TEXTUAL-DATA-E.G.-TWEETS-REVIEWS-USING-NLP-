import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter

# NLTK downloads
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- Sample Dataset ---
data = {
    'text': [
        "Amazing product, loved it!",
        "Terrible service, very disappointed.",
        "It was okay, not bad.",
        "Fantastic support team!",
        "Worst experience ever.",
        "Average item, nothing special.",
        "Super happy with the results!",
        "Completely useless, not worth it.",
        "Neutral review here.",
        "Everything went well, great buy!"
    ],
    'sentiment': [
        'positive', 'negative', 'neutral', 'positive', 'negative',
        'neutral', 'positive', 'negative', 'neutral', 'positive'
    ]
}
df = pd.DataFrame(data)

# --- Preprocessing ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = tokenizer.tokenize(text)
    return " ".join(lemmatizer.lemmatize(w) for w in tokens if w not in stop_words)

df["cleaned_text"] = df["text"].apply(preprocess)

# --- TF-IDF Vectorization ---
X = df["cleaned_text"]
y = df["sentiment"]
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=3, stratify=y, random_state=42
)

# --- Model Training ---
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)

# --- Evaluation ---
y_pred = model.predict(X_test)

print("\n--- Model Evaluation ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Prediction on New Samples ---
def analyze_sentiment(text):
    cleaned = preprocess(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

print("\n--- Predictions on New Text ---")
samples = [
    "I am really satisfied with this item.",
    "The support was horrible, not helpful at all.",
    "It's fine, just average.",
    "Loved the service and quick delivery.",
    "Totally not worth the money."
]

results = []
for s in samples:
    sentiment = analyze_sentiment(s)
    results.append(sentiment)
    print(f"'{s}' -> {sentiment}")

# --- Bar Graph of Predictions ---
sentiment_counts = Counter(results)
plt.figure(figsize=(6, 4))
plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['green', 'red', 'gray'])
plt.title("Sentiment Prediction Summary")
plt.xlabel("Sentiment")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Original Dataset Sentiment Bar Chart ---
plt.figure(figsize=(6, 4))
df['sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'gray'])
plt.title("Original Dataset Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("\n--- Sentiment Analysis Completed ---")
