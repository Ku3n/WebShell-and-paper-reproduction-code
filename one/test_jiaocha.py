import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse

# 设置随机种子确保结果可复现
def set_random_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 定义GRU模型
class GRUModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, output_size=2, dropout=0.05):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.dropout(h_n.squeeze(0))
        out = self.fc(out)
        return out

# 定义WebShell数据集
class WebShellDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        
        # 读取文件内容
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            content = ""
        
        # 移除注释
        if file_path.endswith('.php'):
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
            content = re.sub(r'//.*?$', '', content, flags=re.M)
            content = re.sub(r'#.*?$', '', content, flags=re.M)
        elif file_path.endswith('.jsp'):
            content = re.sub(r'<%--.*?--%>', '', content, flags=re.DOTALL)
            content = re.sub(r'//.*?$', '', content, flags=re.M)
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # 分词
        encoding = self.tokenizer.encode_plus(
            content,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'file_path': file_path
        }

# 训练模型
def train_model(model, train_loader, val_loader, codebert, device, epochs=10, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_f1 = 0.0
    patience = 3
    counter = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [训练]"):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            with torch.no_grad():
                outputs = codebert(inputs, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state
            
            optimizer.zero_grad()
            logits = model(embeddings)
            loss = criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 记录训练损失和预测结果
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # 计算训练指标
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)
        
        # 验证阶段
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, codebert, device)
        
        # 打印训练进度
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"训练: 损失={train_loss/len(train_loader):.4f}, 准确率={train_acc:.4f}, F1={train_f1:.4f}")
        print(f"验证: 损失={val_loss:.4f}, 准确率={val_acc:.4f}, F1={val_f1:.4f}")
        
        # 早停策略
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')
            counter = 0
            print("保存最佳模型")
        else:
            counter += 1
            if counter >= patience:
                print(f"早停触发，在第 {epoch+1} 轮后停止训练")
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    print(f"训练完成，最佳验证F1分数: {best_val_f1:.4f}")
    return model

# 评估模型
def evaluate_model(model, dataloader, codebert, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = codebert(inputs, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state
            logits = model(embeddings)
            loss = criterion(logits, labels)
            
            # 记录损失和预测结果
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算评估指标
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return total_loss/len(dataloader), acc, f1

# 评估测试集并计算指标
def evaluate_testset(model, dataloader, codebert, device, language):
    model.eval()
    all_preds = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            paths = batch['file_path']
            
            # 前向传播
            outputs = codebert(inputs, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state
            logits = model(embeddings)
            
            # 记录预测结果
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print(f"\n{language}文件测试集性能:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1值: {f1:.4f}")
    
    # 保存错误预测的文件
    errors = [
        (path, "误报" if true == 0 and pred == 1 else "漏报")
        for path, true, pred in zip(all_paths, all_labels, all_preds)
        if true != pred
    ]
    
    if errors:
        print(f"{language}文件中有 {len(errors)} 个预测错误")
        pd.DataFrame(errors, columns=['文件路径', '错误类型']).to_csv(f'{language.lower()}_errors.csv', index=False)
        print(f"{language}错误详情已保存至 {language.lower()}_errors.csv")
    else:
        print(f"{language}文件全部预测正确！")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WebShell检测与评估工具')
    parser.add_argument('--normal_php', type=str, help='正常PHP文件目录')
    parser.add_argument('--malicious_php', type=str, help='恶意PHP文件目录')
    parser.add_argument('--normal_jsp', type=str, default='./jsp/normal', help='正常JSP文件目录')
    parser.add_argument('--malicious_jsp', type=str, default='./jsp/shell', help='恶意JSP文件目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_random_seed()
    
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载分词器和CodeBERT模型
    print("加载CodeBERT模型...")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    codebert = AutoModel.from_pretrained('microsoft/codebert-base').to(device)
    
    # 冻结CodeBERT参数
    for param in codebert.parameters():
        param.requires_grad = False
    
    # 收集所有文件
    all_files = []
    
    # 收集PHP文件
    print("收集PHP文件...")
    normal_php_files = [
        (os.path.join(args.normal_php, f), 0) 
        for f in os.listdir(args.normal_php) 
        if f.endswith('.php')
    ]
    malicious_php_files = [
        (os.path.join(args.malicious_php, f), 1) 
        for f in os.listdir(args.malicious_php) 
        if f.endswith('.php')
    ]
    php_files = normal_php_files + malicious_php_files
    all_files.extend(php_files)
    print(f"共收集到 {len(php_files)} 个PHP文件")
    
    # 收集JSP文件
    print("收集JSP文件...")
    normal_jsp_files = [
        (os.path.join(args.normal_jsp, f), 0) 
        for f in os.listdir(args.normal_jsp) 
        if f.endswith('.jsp')
    ]
    malicious_jsp_files = [
        (os.path.join(args.malicious_jsp, f), 1) 
        for f in os.listdir(args.malicious_jsp) 
        if f.endswith('.jsp')
    ]
    jsp_files = normal_jsp_files + malicious_jsp_files
    all_files.extend(jsp_files)
    print(f"共收集到 {len(jsp_files)} 个JSP文件")
    
    # 划分训练集和测试集 (8:2比例)
    print(f"按 {1-args.test_size}:{args.test_size} 比例划分训练集和测试集...")
    train_files, test_files = train_test_split(
        all_files, 
        test_size=args.test_size, 
        random_state=42,
        stratify=[label for _, label in all_files]
    )
    
    print(f"训练集大小: {len(train_files)}")
    print(f"测试集大小: {len(test_files)}")
    
    # 创建数据集和数据加载器
    train_dataset = WebShellDataset(train_files, tokenizer, args.max_length)
    test_dataset = WebShellDataset(test_files, tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 初始化模型
    model = GRUModel().to(device)
    
    # 训练模型
    print("\n开始训练模型...")
    model = train_model(model, train_loader, test_loader, codebert, device, args.epochs, args.lr)
    
    # 分别评估PHP和JSP文件
    print("\n=== 评估结果 ===")
    
    # 评估PHP文件
    php_test_files = [f for f in test_files if f[0].endswith('.php')]
    if php_test_files:
        php_test_dataset = WebShellDataset(php_test_files, tokenizer, args.max_length)
        php_test_loader = DataLoader(php_test_dataset, batch_size=args.batch_size)
        php_metrics = evaluate_testset(model, php_test_loader, codebert, device, "PHP")
    else:
        print("测试集中没有PHP文件")
        php_metrics = None
    
    # 评估JSP文件
    jsp_test_files = [f for f in test_files if f[0].endswith('.jsp')]
    if jsp_test_files:
        jsp_test_dataset = WebShellDataset(jsp_test_files, tokenizer, args.max_length)
        jsp_test_loader = DataLoader(jsp_test_dataset, batch_size=args.batch_size)
        jsp_metrics = evaluate_testset(model, jsp_test_loader, codebert, device, "JSP")
    else:
        print("测试集中没有JSP文件")
        jsp_metrics = None
    
    # 保存总体结果
    if php_metrics and jsp_metrics:
        metrics_df = pd.DataFrame({
            'Language': ['PHP', 'JSP'],
            'Accuracy': [php_metrics['accuracy'], jsp_metrics['accuracy']],
            'Precision': [php_metrics['precision'], jsp_metrics['precision']],
            'Recall': [php_metrics['recall'], jsp_metrics['recall']],
            'F1': [php_metrics['f1'], jsp_metrics['f1']]
        })
        metrics_df.to_csv('webshell_metrics.csv', index=False)
        print("\n所有评估指标已保存至 webshell_metrics.csv")    