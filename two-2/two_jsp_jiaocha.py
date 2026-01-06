import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import re
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebshellDetector:
    def __init__(self, n_gram_range=(1, 4), feature_size=100, model_type='word2vec'):
        self.n_gram_range = n_gram_range
        self.feature_size = feature_size
        self.model_type = model_type
        self.vectorizer = None
        self.scaler = StandardScaler()
        self.model = SVC(probability=True, kernel='rbf', C=10, gamma='scale')

    def read_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"读取文件失败: {file_path}, 错误: {e}")
            return ""

    def extract_ngrams(self, text, n):
        text = re.sub(r'\s+', ' ', text)
        return [text[i:i+n] for i in range(len(text)-n+1)]

    def preprocess_file(self, file_path):
        content = self.read_file(file_path)
        if not content:
            return []
        all_ngrams = []
        for n in range(self.n_gram_range[0], self.n_gram_range[1]+1):
            all_ngrams.extend(self.extract_ngrams(content, n))
        return all_ngrams

    def load_files(self, directory, label):
        file_paths = [os.path.join(dp, f) for dp, _, filenames in os.walk(directory) for f in filenames if f.endswith(('.php', '.jsp'))]
        with ThreadPoolExecutor(max_workers=8) as executor:
            features_list = list(executor.map(self.preprocess_file, file_paths))
        valid_indices = [i for i, features in enumerate(features_list) if features]
        return [features_list[i] for i in valid_indices], [label]*len(valid_indices)

    def create_word2vec_features(self, features_list):
        self.vectorizer = Word2Vec(sentences=features_list, vector_size=self.feature_size, window=5, min_count=1, workers=4)
        features = []
        for tokens in features_list:
            vecs = [self.vectorizer.wv[token] for token in tokens if token in self.vectorizer.wv]
            features.append(np.mean(vecs, axis=0) if vecs else np.zeros(self.feature_size))
        return np.array(features)

    def evaluate(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }

def main():
    benign_dir = 'D:/DeepLearning/otherpaper/two-2/jsp/normal'
    malicious_dir = 'D:/DeepLearning/otherpaper/two-2/jsp/shell'

    detector = WebshellDetector(model_type='word2vec', feature_size=200)

    benign_features, benign_labels = detector.load_files(benign_dir, 0)
    malicious_features, malicious_labels = detector.load_files(malicious_dir, 1)

    all_features = benign_features + malicious_features
    all_labels = benign_labels + malicious_labels

    X = detector.create_word2vec_features(all_features)
    y = np.array(all_labels)

    X = detector.scaler.fit_transform(X)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Fold {fold+1}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = SVC(probability=True, kernel='rbf', C=10, gamma='scale')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = detector.evaluate(y_test, y_pred)
        results.append(metrics)
        logger.info(f"Fold {fold+1} 结果: {metrics}")

    avg_result = pd.DataFrame(results).mean().to_dict()
    logger.info(f"5折交叉验证平均结果: {avg_result}")
    pd.DataFrame(results).to_csv('crossval_metrics.csv', index=False)

if __name__ == '__main__':
    main()
