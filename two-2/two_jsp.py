import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
import re
import joblib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebshellDetector:
    def __init__(self, n_gram_range=(1, 4), feature_size=100, model_type='word2vec'):
        """
        初始化Webshell检测器
        
        Args:
            n_gram_range: N-gram范围
            feature_size: 特征向量维度
            model_type: 特征提取模型类型('tfidf'或'word2vec')
        """
        self.n_gram_range = n_gram_range
        self.feature_size = feature_size
        self.model_type = model_type
        self.vectorizer = None
        self.scaler = StandardScaler()
        self.model = SVC(probability=True, kernel='rbf', C=10, gamma='scale')
        
    def read_file(self, file_path):
        """读取文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"读取文件 {file_path} 失败: {e}")
            return ""
    
    def extract_ngrams(self, text, n):
        """提取文本的N-gram"""
        text = re.sub(r'\s+', ' ', text)  # 替换连续空格
        return [text[i:i+n] for i in range(len(text)-n+1)]
    
    def preprocess_file(self, file_path):
        """预处理文件内容，提取特征"""
        content = self.read_file(file_path)
        if not content:
            return []
        
        # 提取N-gram
        all_ngrams = []
        for n in range(self.n_gram_range[0], self.n_gram_range[1]+1):
            all_ngrams.extend(self.extract_ngrams(content, n))
        
        return all_ngrams
    
    def load_files(self, directory, label):
        """加载目录下的所有文件"""
        logger.info(f"加载目录 {directory} 中的文件，标签: {label}")
        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.php', '.jsp')):
                    file_paths.append(os.path.join(root, file))
        
        # 并行处理文件
        with ThreadPoolExecutor(max_workers=8) as executor:
            features_list = list(executor.map(self.preprocess_file, file_paths))
        
        # 过滤空特征
        valid_indices = [i for i, features in enumerate(features_list) if features]
        valid_features = [features_list[i] for i in valid_indices]
        labels = [label] * len(valid_features)
        
        logger.info(f"成功加载 {len(valid_features)} 个有效文件")
        return valid_features, labels, [file_paths[i] for i in valid_indices]
    
    def create_tfidf_features(self, features_list):
        """使用TF-IDF创建特征向量"""
        logger.info("使用TF-IDF提取特征")
        # 将特征列表转换为文本格式
        texts = [' '.join(features) for features in features_list]
        
        # 初始化并拟合TF-IDF向量器
        self.vectorizer = TfidfVectorizer(ngram_range=self.n_gram_range, max_features=self.feature_size)
        return self.vectorizer.fit_transform(texts)
    
    def create_word2vec_features(self, features_list):
        """使用Word2Vec创建特征向量"""
        logger.info("使用Word2Vec提取特征")
        # 训练Word2Vec模型
        self.vectorizer = Word2Vec(sentences=features_list, vector_size=self.feature_size, 
                                  window=5, min_count=1, workers=4)
        
        # 为每个文件创建特征向量
        features = []
        for tokens in features_list:
            if not tokens:
                features.append(np.zeros(self.feature_size))
                continue
                
            # 计算所有token的向量平均值
            vecs = [self.vectorizer.wv[token] for token in tokens if token in self.vectorizer.wv]
            if not vecs:
                features.append(np.zeros(self.feature_size))
            else:
                features.append(np.mean(vecs, axis=0))
        
        return np.array(features)
    
    def fit(self, features_list, labels):
        """训练模型"""
        logger.info(f"开始使用{self.model_type}训练模型")
        
        # 提取特征
        if self.model_type == 'tfidf':
            X = self.create_tfidf_features(features_list).toarray()
        elif self.model_type == 'word2vec':
            X = self.create_word2vec_features(features_list)
        else:
            raise ValueError("model_type必须是'tfidf'或'word2vec'")
        
        y = np.array(labels)
        
        # 标准化特征
        X = self.scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        logger.info(f"训练集类别分布: {Counter(y_train)}")
        logger.info(f"测试集类别分布: {Counter(y_test)}")
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 在测试集上评估
        y_pred = self.model.predict(X_test)
        self.evaluate(y_test, y_pred)
        
        return self
    
    def evaluate(self, y_true, y_pred):
        """评估模型性能"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        
        logger.info("模型评估结果:")
        logger.info(f"准确率: {accuracy:.4f}")
        logger.info(f"精确率: {precision:.4f}")
        logger.info(f"召回率: {recall:.4f}")
        logger.info(f"F1分数: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def predict(self, file_path):
        """预测单个文件是否为Webshell"""
        features = self.preprocess_file(file_path)
        if not features:
            return 0  # 默认预测为非Webshell
        
        if self.model_type == 'tfidf':
            text = ' '.join(features)
            X = self.vectorizer.transform([text]).toarray()
        elif self.model_type == 'word2vec':
            vecs = [self.vectorizer.wv[token] for token in features if token in self.vectorizer.wv]
            if not vecs:
                X = np.zeros((1, self.feature_size))
            else:
                X = np.mean(vecs, axis=0).reshape(1, -1)
        
        X = self.scaler.transform(X)
        return self.model.predict(X)[0]
    
    def save_model(self, model_path):
        """保存模型"""
        model_data = {
            'vectorizer': self.vectorizer,
            'scaler': self.scaler,
            'model': self.model,
            'n_gram_range': self.n_gram_range,
            'feature_size': self.feature_size,
            'model_type': self.model_type
        }
        joblib.dump(model_data, model_path)
        logger.info(f"模型已保存到 {model_path}")
    
    @staticmethod
    def load_model(model_path):
        """加载模型"""
        model_data = joblib.load(model_path)
        detector = WebshellDetector(
            n_gram_range=model_data['n_gram_range'],
            feature_size=model_data['feature_size'],
            model_type=model_data['model_type']
        )
        detector.vectorizer = model_data['vectorizer']
        detector.scaler = model_data['scaler']
        detector.model = model_data['model']
        return detector

def main():
    # 设置数据路径
    benign_dir = 'D:/DeepLearning/otherpaper/two-2/jsp/normal'  # 良性文件目录
    malicious_dir = 'D:/DeepLearning/otherpaper/two-2/jsp/shell'  # 恶意文件目录
    
    # 创建检测器实例
    detector = WebshellDetector(model_type='word2vec', feature_size=200)
    
    # 加载数据
    benign_features, benign_labels, benign_files = detector.load_files(benign_dir, 0)
    malicious_features, malicious_labels, malicious_files = detector.load_files(malicious_dir, 1)
    
    # 合并数据集
    all_features = benign_features + malicious_features
    all_labels = benign_labels + malicious_labels
    all_files = benign_files + malicious_files
    
    # 训练模型
    detector.fit(all_features, all_labels)
    

if __name__ == "__main__":
    main()    