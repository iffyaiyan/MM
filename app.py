import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from sklearn.metrics import f1_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the label list
label_list = ['O', 'B-Treatment', 'I-Treatment', 'B-Chronic_disease', 'I-Chronic_disease', 'B-Cancer', 'I-Cancer', 'B-Allergy_name', 'I-Allergy_name']

# Tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

class NERDataset(Dataset):
    def __init__(self, texts, tags):
        self.texts = texts
        self.tags = tags
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tag = self.tags[idx]
        
        encoding = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        
        # Process tags
        words = text.split()
        bio_tags = ['O'] * len(words)
        for tag_info in tag.split(','):
            if tag_info:
                parts = tag_info.split(':')
                if len(parts) == 3:
                    start, end, label = parts
                    try:
                        start, end = int(start), int(end)
                        if start < len(bio_tags):
                            bio_tags[start] = f'B-{label.capitalize()}'
                            for i in range(start + 1, min(end, len(bio_tags))):
                                bio_tags[i] = f'I-{label.capitalize()}'
                    except ValueError:
                        logger.warning(f"Invalid start or end index in tag: {tag_info}")
                else:
                    logger.warning(f"Unexpected tag format: {tag_info}")
        
        # Align tags with tokens
        aligned_tags = []
        word_ids = encoding.word_ids()
        for word_id in word_ids:
            if word_id is None:
                aligned_tags.append('O')
            elif word_id < len(bio_tags):
                aligned_tags.append(bio_tags[word_id])
            else:
                aligned_tags.append('O')
        
        # Convert tags to ids
        label_ids = [label_list.index(t) if t in label_list else 0 for t in aligned_tags]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids)
        }

def prepare_data(file_path):
    df = pd.read_csv(file_path)
    texts = df['text'].tolist()
    tags = df['tags'].tolist()
    
    train_texts, test_texts, train_tags, test_tags = train_test_split(texts, tags, test_size=0.2, random_state=42)
    train_texts, val_texts, train_tags, val_tags = train_test_split(train_texts, train_tags, test_size=0.1, random_state=42)
    
    return (
        NERDataset(train_texts, train_tags),
        NERDataset(val_texts, val_tags),
        NERDataset(test_texts, test_tags)
    )

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            pred = torch.argmax(logits, dim=2)
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return predictions, true_labels

def calculate_f1_scores(predictions, true_labels):
    pred_flat = np.concatenate(predictions)
    true_flat = np.concatenate(true_labels)
    
    # Filter out padding tokens
    mask = true_flat != -100
    pred_flat = pred_flat[mask]
    true_flat = true_flat[mask]
    
    # Convert to binary format
    mlb = MultiLabelBinarizer()
    true_binary = mlb.fit_transform([[label] for label in true_flat])
    pred_binary = mlb.transform([[label] for label in pred_flat])
    
    # Calculate F1 scores
    f1_scores = {}
    for i, label in enumerate(mlb.classes_):
        if label == 0:  # Skip 'O' label
            continue
        true_label = true_binary[:, i]
        pred_label = pred_binary[:, i]
        f1 = f1_score(true_label, pred_label, average='binary', zero_division=0)
        f1_scores[label_list[label]] = f1
    
    # Calculate weighted average F1
    weights = np.sum(true_binary, axis=0)
    weighted_f1 = np.sum([f1_scores[label_list[i]] * weights[i] for i in range(1, len(weights))]) / np.sum(weights[1:])
    f1_scores['Weighted Average'] = weighted_f1
    
    return f1_scores

def train_continual(model, optimizer, dataloaders, num_epochs, device):
    for task, (train_loader, val_loader, test_loader) in enumerate(dataloaders):
        print(f"Training on Task {task+1}")
        
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")
            
            # Evaluate on validation set
            val_predictions, val_true_labels = evaluate(model, val_loader, device)
            val_f1_scores = calculate_f1_scores(val_predictions, val_true_labels)
            print("Validation F1 Scores:", val_f1_scores)
        
        # Evaluate on all seen tasks
        for seen_task, (_, _, seen_test_loader) in enumerate(dataloaders[:task+1]):
            test_predictions, test_true_labels = evaluate(model, seen_test_loader, device)
            test_f1_scores = calculate_f1_scores(test_predictions, test_true_labels)
            print(f"Test F1 Scores for Task {seen_task+1}:", test_f1_scores)

def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare datasets
    G1_train, G1_val, G1_test = prepare_data('G1.csv')
    G2_train, G2_val, G2_test = prepare_data('G2.csv')
    G3_train, G3_val, G3_test = prepare_data('G3.csv')
    
    # Create model
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(label_list))
    model.to(device)
    
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Create dataloaders
    batch_size = 32
    dataloaders = [
        (DataLoader(G1_train, batch_size=batch_size, shuffle=True),
         DataLoader(G1_val, batch_size=batch_size),
         DataLoader(G1_test, batch_size=batch_size)),
        (DataLoader(G2_train, batch_size=batch_size, shuffle=True),
         DataLoader(G2_val, batch_size=batch_size),
         DataLoader(G2_test, batch_size=batch_size)),
        (DataLoader(G3_train, batch_size=batch_size, shuffle=True),
         DataLoader(G3_val, batch_size=batch_size),
         DataLoader(G3_test, batch_size=batch_size))
    ]
    
    # Train the model
    num_epochs = 7
    train_continual(model, optimizer, dataloaders, num_epochs, device)
    
    # Save the final model
    torch.save(model.state_dict(), 'continual_ner_model.pth')

if __name__ == '__main__':
    main()