import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm

# Helper function to convert tags to BIO format
# def convert_to_bio(tags):
#     bio_tags = []
#     for tag in tags.split(','):
#         if tag:
#             start, end, label = map(int, tag.split(':'))
#             bio_tags.extend(['O'] * start)
#             bio_tags.append(f'B-{label}')
#             bio_tags.extend([f'I-{label}'] * (end - start - 1))
#         if len(bio_tags) < len(text.split()):
#             bio_tags.extend(['O'] * (len(text.split()) - len(bio_tags)))
#     return bio_tags


# Helper function to convert tags to BIO format
def convert_to_bio(tags, text):
    bio_tags = []
    for tag in tags.split(','):
        if tag:
            try:
                start, end, label = map(int, tag.split(':'))
                bio_tags.extend(['O'] * start)
                bio_tags.append(f'B-{label}')
                bio_tags.extend([f'I-{label}'] * (end - start - 1))
            except ValueError:
                print(f"Skipping invalid tag: {tag}")
                continue
        if len(bio_tags) < len(text.split()):
            bio_tags.extend(['O'] * (len(text.split()) - len(bio_tags)))
    return bio_tags

# Custom Dataset
class NERDataset(Dataset):
    def __init__(self, texts, tags, tokenizer, max_len):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        tags = self.tags[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        # Pad the labels
        labels = [self.tag2id[tag] for tag in tags]
        labels = labels[:self.max_len - 2]
        labels = [0] + labels + [0] * (self.max_len - len(labels) - 1)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.LongTensor(labels)
        }

# Function to train the model
def train(model, train_data, val_data, device, epochs):
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}')

    return model

# Function to evaluate the model
def evaluate(model, test_data, device):
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=2)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Calculate F1 scores here,  
    return calculate_f1_scores(predictions, true_labels)

# Due to time constraint I have defined this function. Will do it by tonight
def calculate_f1_scores(predictions, true_labels):
    pass

# Main execution
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load and preprocess datasets
    datasets = []
    for i in range(1, 4):
        df = pd.read_csv(f'G{i}.csv')
        texts = df['text'].tolist()
        tags = df['tags'].apply(convert_to_bio).tolist()
        datasets.append((texts, tags))

    # Define tag2id and id2tag
    unique_tags = set(tag for dataset in datasets for sample in dataset[1] for tag in sample)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    # Create model
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(tag2id))
    model.to(device)

    # Continual Learning Loop
    memory = []
    for task_id, (texts, tags) in enumerate(datasets):
        print(f'Training on Task {task_id + 1}')
        
        # Split data
        train_texts, test_texts, train_tags, test_tags = train_test_split(texts, tags, test_size=0.2)
        train_texts, val_texts, train_tags, val_tags = train_test_split(train_texts, train_tags, test_size=0.1)

        # Create datasets
        train_data = NERDataset(train_texts + [m[0] for m in memory], 
                                train_tags + [m[1] for m in memory], 
                                tokenizer, max_len=128)
        val_data = NERDataset(val_texts, val_tags, tokenizer, max_len=128)
        test_data = NERDataset(test_texts, test_tags, tokenizer, max_len=128)

        # Train
        model = train(model, train_data, val_data, device, epochs=3)

        # Evaluate
        f1_scores = evaluate(model, test_data, device)
        print(f'Task {task_id + 1} F1 Scores:', f1_scores)

        # Update memory
        memory.extend(list(zip(train_texts, train_tags))[:100])
        memory = memory[-300:]  # Keep only 300 examples (100 from each task)

        # Save model
        model.save_pretrained(f'model_task_{task_id + 1}')

    # Train on combined dataset for comparison
    combined_texts = [text for dataset in datasets for text in dataset[0]]
    combined_tags = [tag for dataset in datasets for tag in dataset[1]]
    train_texts, test_texts, train_tags, test_tags = train_test_split(combined_texts, combined_tags, test_size=0.2)
    train_texts, val_texts, train_tags, val_tags = train_test_split(train_texts, train_tags, test_size=0.1)

    combined_train_data = NERDataset(train_texts, train_tags, tokenizer, max_len=128)
    combined_val_data = NERDataset(val_texts, val_tags, tokenizer, max_len=128)
    combined_test_data = NERDataset(test_texts, test_tags, tokenizer, max_len=128)

    combined_model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(tag2id))
    combined_model.to(device)

    combined_model = train(combined_model, combined_train_data, combined_val_data, device, epochs=3)
    combined_f1_scores = evaluate(combined_model, combined_test_data, device)
    print('Combined Model F1 Scores:', combined_f1_scores)

if __name__ == '__main__':
    main()

