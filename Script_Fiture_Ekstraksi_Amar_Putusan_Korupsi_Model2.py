import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np

# Step 1: Load and Clean the Dataset
print("Loading dataset...")
df = pd.read_csv('filtered_sampled_data.csv')  # Replace with your file path 

# Fill missing values and ensure text is clean for the specified columns
columns_to_clean = [
    'amar', 'klasifikasi', 'lama_hukuman', 'lembaga_peradilan', 'provinsi', 'status',
    'sub_klasifikasi', 'url', 'kepala_putusan', 'amar_putusan', 'riwayat_penahanan',
    'riwayat_perkara', 'riwayat_tuntutan', 'riwayat_dakwaan', 'fakta', 'penutup'
]

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    return text

# Apply filling missing values and text cleaning to specified columns
print("Cleaning text data...")
for column in columns_to_clean:
    df[column] = df[column].fillna('').astype(str)
    df[column] = df[column].apply(clean_text)

print("Data cleaning complete.")

# Step 2: Check Class Distribution
print("Plotting class distribution...")
plt.figure(figsize=(12, 6))
sns.countplot(df['sub_klasifikasi'])
plt.title('Class Distribution')
plt.xticks(rotation=90)
plt.show()

# Step 3: Train-Test Split and Tokenization
print("Splitting data into train, validation, and test sets...")
# Split the data into train, validation, and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['amar_putusan'], df['sub_klasifikasi'], test_size=0.2, random_state=42)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.1, random_state=42)

# Load the tokenizer
print("Loading tokenizer...")
tokenizer = BertTokenizerFast.from_pretrained('indobenchmark/indobert-base-p1')

# Define the maximum length for tokenization
max_length = 512

# Tokenize the data
print("Tokenizing data...")
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='pt')
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='pt')
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='pt')

# Ensure the token type IDs tensor is handled correctly
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_labels = train_labels.astype('category').cat.codes.tolist()
val_labels = val_labels.astype('category').cat.codes.tolist()
test_labels = test_labels.astype('category').cat.codes.tolist()

train_dataset = CustomDataset(train_encodings, train_labels)
val_dataset = CustomDataset(val_encodings, val_labels)
test_dataset = CustomDataset(test_encodings, test_labels)

print("Dataset conversion complete.")

# Create DataLoader
print("Creating data loaders...")
train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=16)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=16)
test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=16)

# Set device to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# Step 5: Load the Model
print("Loading model...")
model = BertForSequenceClassification.from_pretrained(
    'indobenchmark/indobert-base-p1', 
    num_labels=len(set(train_labels)),
    ignore_mismatched_sizes=True
)

# Move model to device (GPU)
model.to(device)
print("Model loaded and moved to device.")

# Set up the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_loader) * 3  # 3 is the number of epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Step 6: Training Loop
def train_epoch(model, data_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return total_loss / len(data_loader)

def eval_model(model, data_loader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    return avg_loss, accuracy, precision, recall, f1, all_preds

# Save best model
best_f1 = 0

num_epochs = 5
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train_epoch(model, train_loader, optimizer, scheduler)
    print(f'Train Loss: {train_loss:.4f}')
    val_loss, val_accuracy, val_precision, val_recall, val_f1, _ = eval_model(model, val_loader)
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')
    if val_f1 > best_f1:
        best_f1 = val_f1
        print("Saving best model...")
        torch.save(model.state_dict(), 'best_model.pt')

# Load the best model for testing
print("Loading best model for testing...")
model.load_state_dict(torch.load('best_model.pt'))

# Step 7: Test the Model
print("Evaluating model on test data...")
test_loss, test_accuracy, test_precision, test_recall, test_f1, all_preds = eval_model(model, test_loader)
print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')

# Step 8: Evaluate 'korupsi' Class
# Extracting precision, recall, and F1 for 'korupsi' class
print("Evaluating 'korupsi' class performance...")
test_labels_np = np.array(test_labels)
all_preds_np = np.array(all_preds)
korupsi_index = df['sub_klasifikasi'].astype('category').cat.categories.tolist().index('korupsi')

korupsi_precision, korupsi_recall, korupsi_f1, _ = precision_recall_fscore_support(
    test_labels_np, all_preds_np, labels=[korupsi_index], average='macro', zero_division=0)

print(f'Korupsi Class - Precision: {korupsi_precision:.4f}, Recall: {korupsi_recall:.4f}, F1: {korupsi_f1:.4f}')
