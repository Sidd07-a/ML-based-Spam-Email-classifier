import os
import re
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

nltk.download('stopwords')

try:
    df = pd.read_csv("email_spam_dataset_balanced_15000.csv", encoding="latin-1", on_bad_lines="skip", low_memory=False)
    print(" Dataset loaded successfully!")
except Exception as e:
    print(" Error loading dataset:", e)
    exit()

print("\nColumns in dataset:", list(df.columns))

possible_text_cols = ['message', 'text', 'content', 'body', 'Message']
text_col = next((c for c in possible_text_cols if c in df.columns), None)

if not text_col:
    print(" No text column found! Check your CSV column names.")
    exit()

if 'label' not in df.columns:
    print(" No label column found — creating dummy labels for testing.")
    df['label'] = np.random.randint(0, 2, df.shape[0])

df = df[[text_col, 'label']]
df.columns = ['message', 'label']
df.dropna(subset=['message'], inplace=True)
df['message'] = df['message'].astype(str)

df['label'] = df['label'].map({'ham': 0, 'spam': 1})
print("\nLabel distribution:\n", df['label'].value_counts())

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = [ps.stem(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

print("\n Cleaning text...")
tqdm.pandas(desc="Cleaning progress")
df['cleaned'] = df['message'].progress_apply(clean_text)
df = df[df['cleaned'].str.strip() != ""]

vectorizer = TfidfVectorizer(max_features=7000, stop_words='english')
X = vectorizer.fit_transform(df['cleaned']).toarray()

scaler = StandardScaler(with_mean=False)
X = scaler.fit_transform(X)

y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

model = LogisticRegressionModel(X_train.shape[1]).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 30
print("\n Training model (with BCEWithLogitsLoss)...")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 2 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float().cpu().numpy()
    y_true = y_test.cpu().numpy()

acc = accuracy_score(y_true, preds)
print("\n Model Evaluation:")
print("Accuracy:", round(acc * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_true, preds, target_names=['HAM', 'SPAM']))
print("\nConfusion Matrix:\n", confusion_matrix(y_true, preds))

torch.save(model.state_dict(), "spam_model_gpu.pth")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(scaler, "tfidf_scaler.pkl")
print("\n Model, vectorizer, and scaler saved successfully!")

def predict_email(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned]).toarray()
    vector = scaler.transform(vector)
    vector = torch.tensor(vector, dtype=torch.float32).to(device)
    with torch.no_grad():
        logit = model(vector)
        prob = torch.sigmoid(logit).item()
    if prob >= 0.5:
        return f" SPAM (Confidence: {prob*100:.2f}%)"
    else:
        return f" HAM (Confidence: {(1-prob)*100:.2f}%)"

print("\n---  Test Your Own Emails ---")
while True:
    user_input = input("Enter email text (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    print("Prediction:", predict_email(user_input))
    print("-" * 50)
