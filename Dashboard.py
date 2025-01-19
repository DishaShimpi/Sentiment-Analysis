import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from tqdm import tqdm

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Preprocessing Function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters and punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Dataset Class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0).to(device),
            "attention_mask": encoding["attention_mask"].squeeze(0).to(device),
            "label": torch.tensor(label, dtype=torch.long).to(device),
        }

# Load Dataset
DATA_PATH = "processed_dataset.csv"
df = pd.read_csv(DATA_PATH)
df['Comments'] = df['Comments'].fillna('missing').apply(preprocess_text)
df['Sentiment'] = df['Sentiment'].fillna('neutral')

print("Dataset Head:\n", df.head())

# Label Encoding
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Sentiment'])
num_labels = len(label_encoder.classes_)
print(f"Number of unique labels: {num_labels} ({label_encoder.classes_})")

# Label Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Sentiment', order=label_encoder.classes_)
plt.title("Label Distribution")
plt.show()

# Split Data
from sklearn.model_selection import train_test_split
MAX_LEN = 128
BATCH_SIZE = 16

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['Comments'], df['label'], test_size=0.2, random_state=42
)

# Tokenizer
MODEL_NAME = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

# Datasets and Dataloaders
train_dataset = SentimentDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, MAX_LEN)
test_dataset = SentimentDataset(test_texts.tolist(), test_labels.tolist(), tokenizer, MAX_LEN)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize and Load Pre-Trained Model
pretrained_model_path = "distilbert_sentiment_model.pt"

model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).to(device)
model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
model.eval()
print("Pre-trained model loaded successfully.")

# Evaluation Function
def evaluate(model, test_loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds = []
    all_labels = []
    all_probs = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["label"]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
    print("\nConfusion Matrix:")
    print(cm)

    return total_loss / len(test_loader), accuracy, precision, recall, f1, cm, all_probs, all_labels

# Plot Confusion Matrix
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

# Plot ROC Curve
def plot_roc_curve(y_true, y_probs, classes):
    plt.figure(figsize=(10, 8))
    for i, class_label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {class_label} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

# Evaluate the Model
test_loss, test_accuracy, precision, recall, f1, cm, all_probs, all_labels = evaluate(model, test_loader)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# Plot Confusion Matrix
plot_confusion_matrix(cm, label_encoder.classes_)

# Plot ROC Curve
plot_roc_curve(np.array(all_labels), np.array(all_probs), label_encoder.classes_)
