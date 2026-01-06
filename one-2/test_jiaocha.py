import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse


def set_random_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GRUModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, output_size=2, dropout=0.05):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.dropout(h_n.squeeze(0))
        return self.fc(out)


class WebShellDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            content = ""

        if file_path.endswith('.php'):
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
            content = re.sub(r'//.*?$', '', content, flags=re.M)
            content = re.sub(r'#.*?$', '', content, flags=re.M)
        elif file_path.endswith('.jsp'):
            content = re.sub(r'<%--.*?--%>', '', content, flags=re.DOTALL)
            content = re.sub(r'//.*?$', '', content, flags=re.M)
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

        encoding = self.tokenizer.encode_plus(
            content, add_special_tokens=True, max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label),
            'file_path': file_path
        }


def train_model(model, train_loader, val_loader, codebert, device, epochs=10, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_f1 = 0
    for epoch in range(epochs):
        model.train()
        all_preds, all_labels = [], []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with torch.no_grad():
                outputs = codebert(inputs, attention_mask=masks).last_hidden_state

            logits = model(outputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        f1 = f1_score(all_labels, all_preds)
        _, val_acc, val_f1 = evaluate_model(model, val_loader, codebert, device)
        print(f"Epoch {epoch+1}: Train F1={f1:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')

    model.load_state_dict(torch.load('best_model.pth'))
    return model


def evaluate_model(model, loader, codebert, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = codebert(inputs, attention_mask=masks).last_hidden_state
            logits = model(outputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return total_loss / len(loader), acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--normal_php', type=str)
    parser.add_argument('--malicious_php', type=str)
    parser.add_argument('--normal_jsp', type=str, default='./jsp/normal')
    parser.add_argument('--malicious_jsp', type=str, default='./jsp/shell')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    set_random_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    codebert = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
    for p in codebert.parameters():
        p.requires_grad = False

    all_files = []
    for folder, label in [
        (args.normal_php, 0), (args.malicious_php, 1),
        (args.normal_jsp, 0), (args.malicious_jsp, 1)]:
        all_files.extend([(os.path.join(folder, f), label) for f in os.listdir(folder) if f.endswith(('.php', '.jsp'))])

    labels = [label for _, label in all_files]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    results = []

    for train_idx, test_idx in skf.split(all_files, labels):
        print(f"\n==== 第 {fold} 折训练 ====")
        train_files = [all_files[i] for i in train_idx]
        test_files = [all_files[i] for i in test_idx]

        train_dataset = WebShellDataset(train_files, tokenizer, args.max_length)
        test_dataset = WebShellDataset(test_files, tokenizer, args.max_length)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        model = GRUModel().to(device)
        model = train_model(model, train_loader, test_loader, codebert, device, args.epochs, args.lr)

        _, acc, f1 = evaluate_model(model, test_loader, codebert, device)
        precision = precision_score([d[1] for d in test_files], [torch.argmax(model(codebert(tokenizer(d[0], return_tensors='pt')['input_ids'].to(device)).last_hidden_state), dim=1).item() for d in test_files])
        recall = recall_score([d[1] for d in test_files], [torch.argmax(model(codebert(tokenizer(d[0], return_tensors='pt')['input_ids'].to(device)).last_hidden_state), dim=1).item() for d in test_files])

        print(f"Fold-{fold} 准确率: {acc:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")
        results.append({'Fold': fold, 'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1': f1})
        fold += 1

    df = pd.DataFrame(results)
    print("\n==== 5折平均结果 ====")
    print(df.mean(numeric_only=True))
    df.to_csv("crossval_metrics.csv", index=False)


if __name__ == "__main__":
    main()
