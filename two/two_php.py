import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
from tempfile import TemporaryFile
import re
import joblib
from collections import Counter
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryEfficientWebshellDetector:
    def __init__(self, n_gram_range=(1, 4), feature_size=100, model_type='word2vec', batch_size=1000):
        """
        初始化内存高效的Webshell检测器
        
        Args:
            n_gram_range: N-gram范围
            feature_size: 特征向量维度
            model_type: 特征提取模型类型('tfidf'或'word2vec')
            batch_size: 批处理大小
        """
        self.n_gram_range = n_gram_range
        self.feature_size = feature_size
        self.model_type = model_type
        self.batch_size = batch_size
        self.vectorizer = None
        self.scaler = StandardScaler()
        self.dimensionality_reducer = TruncatedSVD(n_components=min(feature_size, 100))  # 用于TF-IDF降维
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
    
    def load_files_generator(self, directory, label):
        """生成器方式加载目录下的所有文件"""
        logger.info(f"加载目录 {directory} 中的文件，标签: {label}")
        count = 0
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.php', '.jsp')):
                    file_path = os.path.join(root, file)
                    features = self.preprocess_file(file_path)
                    if features:
                        yield features, label, file_path
                        count += 1
                        if count % 1000 == 0:
                            logger.info(f"已加载 {count} 个文件")
        
        logger.info(f"成功加载 {count} 个有效文件")
    
    def create_tfidf_features_incremental(self, files_generator):
        """增量式使用TF-IDF创建特征向量"""
        logger.info("使用增量TF-IDF提取特征")
        
        # 第一阶段：拟合词汇表
        self.vectorizer = TfidfVectorizer(ngram_range=self.n_gram_range, max_features=self.feature_size)
        
        # 收集文本样本
        texts = []
        labels = []
        file_paths = []
        
        for features, label, file_path in files_generator:
            texts.append(' '.join(features))
            labels.append(label)
            file_paths.append(file_path)
            
            # 批处理
            if len(texts) >= self.batch_size:
                # 处理当前批次
                if not self.vectorizer:
                    self.vectorizer = TfidfVectorizer(ngram_range=self.n_gram_range, max_features=self.feature_size)
                    self.vectorizer.fit(texts)
                else:
                    # 更新词汇表（简化版本，实际中可能需要更复杂的实现）
                    pass
                
                texts = []
        
        # 最终拟合
        if texts:
            self.vectorizer.fit(texts)
        
        # 第二阶段：转换为特征矩阵
        logger.info("将文本转换为特征矩阵")
        X = None
        y = np.array([])
        all_file_paths = []
        
        texts = []
        current_labels = []
        current_paths = []
        
        for features, label, file_path in files_generator:
            texts.append(' '.join(features))
            current_labels.append(label)
            current_paths.append(file_path)
            
            if len(texts) >= self.batch_size:
                # 处理当前批次
                batch_X = self.vectorizer.transform(texts)
                
                # 降维处理
                if X is None:
                    self.dimensionality_reducer.fit(batch_X)
                    X = self.dimensionality_reducer.transform(batch_X)
                else:
                    batch_X_reduced = self.dimensionality_reducer.transform(batch_X)
                    X = np.vstack((X, batch_X_reduced))
                
                y = np.concatenate((y, np.array(current_labels)))
                all_file_paths.extend(current_paths)
                
                texts = []
                current_labels = []
                current_paths = []
        
        # 处理剩余批次
        if texts:
            batch_X = self.vectorizer.transform(texts)
            batch_X_reduced = self.dimensionality_reducer.transform(batch_X)
            
            if X is None:
                X = batch_X_reduced
            else:
                X = np.vstack((X, batch_X_reduced))
            
            y = np.concatenate((y, np.array(current_labels)))
            all_file_paths.extend(current_paths)
        
        return X, y, all_file_paths
    
    def create_word2vec_features_incremental(self, files_generator):
        """增量式使用Word2Vec创建特征向量"""
        logger.info("使用增量Word2Vec提取特征")
        
        # 第一阶段：训练Word2Vec模型
        self.vectorizer = Word2Vec(vector_size=self.feature_size, window=5, min_count=1, workers=4)
        
        # 存储文件路径以便后续处理
        all_files = []
        
        # 分批次训练Word2Vec
        batch = []
        first_batch = True  # 标记是否是第一个批次
        
        for features, label, file_path in files_generator:
            batch.append(features)
            all_files.append((file_path, label))  # 保存文件路径和标签
            if len(batch) >= self.batch_size:
                # 第一次构建词汇表时使用update=False，后续使用update=True
                self.vectorizer.build_vocab(batch, update=not first_batch)
                self.vectorizer.train(batch, total_examples=len(batch), epochs=self.vectorizer.epochs)
                
                if first_batch:
                    first_batch = False
                
                batch = []
        
        # 处理剩余数据
        if batch:
            self.vectorizer.build_vocab(batch, update=not first_batch)
            self.vectorizer.train(batch, total_examples=len(batch), epochs=self.vectorizer.epochs)
        
        # 第二阶段：转换为特征矩阵
        logger.info("将文本转换为特征矩阵")
        
        # 使用临时文件存储特征向量
        temp_file = TemporaryFile()
        y = []
        all_file_paths = []
        batch_count = 0
        
        # 分批次处理并保存特征向量
        batch_features = []
        batch_labels = []
        batch_paths = []
        
        # 重新处理文件（从保存的路径）
        for file_path, label in all_files:
            features = self.preprocess_file(file_path)
            if not features:
                feature_vector = np.zeros(self.feature_size)
            else:
                # 计算当前文件的特征向量
                vecs = [self.vectorizer.wv[token] for token in features if token in self.vectorizer.wv]
                if not vecs:
                    feature_vector = np.zeros(self.feature_size)
                else:
                    feature_vector = np.mean(vecs, axis=0)
            
            batch_features.append(feature_vector)
            batch_labels.append(label)
            batch_paths.append(file_path)
            
            # 批处理保存
            if len(batch_features) >= self.batch_size:
                batch_array = np.array(batch_features)
                # 保存当前批次到文件
                if batch_count == 0:
                    np.save(temp_file, batch_array)
                else:
                    # 追加模式：先读取已有数据，再合并保存
                    temp_file.seek(0)
                    existing_data = np.load(temp_file)
                    combined_data = np.vstack((existing_data, batch_array))
                    temp_file.seek(0)
                    np.save(temp_file, combined_data)
                
                batch_features = []
                batch_count += 1
        
        # 处理最后一批数据
        if batch_features:
            batch_array = np.array(batch_features)
            if batch_count == 0:
                np.save(temp_file, batch_array)
            else:
                temp_file.seek(0)
                existing_data = np.load(temp_file)
                combined_data = np.vstack((existing_data, batch_array))
                temp_file.seek(0)
                np.save(temp_file, combined_data)
        
        # 读取完整的特征矩阵
        temp_file.seek(0)
        try:
            X = np.load(temp_file)
        except EOFError:
            logger.error("特征矩阵文件为空，可能没有数据被正确保存")
            X = np.array([])
        
        y = np.array(batch_labels)
        all_file_paths = batch_paths
        
        # 关闭临时文件
        temp_file.close()
        
        return X, y, all_file_paths
    
    def fit_incremental(self, files_generator):
        """增量式训练模型"""
        logger.info(f"开始使用{self.model_type}增量训练模型")
        
        # 提取特征
        if self.model_type == 'tfidf':
            X, y, file_paths = self.create_tfidf_features_incremental(files_generator)
        elif self.model_type == 'word2vec':
            X, y, file_paths = self.create_word2vec_features_incremental(files_generator)
        else:
            raise ValueError("model_type必须是'tfidf'或'word2vec'")
        
        logger.info(f"总样本数: {len(y)}")
        logger.info(f"类别分布: {Counter(y)}")
        
        if len(y) == 0:
            logger.error("没有数据用于训练")
            return None
        
        # 标准化特征
        X = self.scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else y
        )
        
        logger.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 在测试集上评估
        y_pred = self.model.predict(X_test)
        metrics = self.evaluate(y_test, y_pred)
        
        return metrics
    
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
            X = self.vectorizer.transform([text])
            X = self.dimensionality_reducer.transform(X)
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
        if hasattr(self, 'dimensionality_reducer'):
            model_data['dimensionality_reducer'] = self.dimensionality_reducer
            
        joblib.dump(model_data, model_path)
        logger.info(f"模型已保存到 {model_path}")
    
    @staticmethod
    def load_model(model_path):
        """加载模型"""
        model_data = joblib.load(model_path)
        detector = MemoryEfficientWebshellDetector(
            n_gram_range=model_data['n_gram_range'],
            feature_size=model_data['feature_size'],
            model_type=model_data['model_type']
        )
        detector.vectorizer = model_data['vectorizer']
        detector.scaler = model_data['scaler']
        detector.model = model_data['model']
        if 'dimensionality_reducer' in model_data:
            detector.dimensionality_reducer = model_data['dimensionality_reducer']
        return detector

def main():
    # 设置数据路径
    benign_dir = 'D:/DeepLearning/otherpaper/two/php/normal'   # 良性文件目录
    malicious_dir = 'D:/DeepLearning/otherpaper/two/php/shell'  # 恶意文件目录
    
    # 创建检测器实例
    detector = MemoryEfficientWebshellDetector(
        model_type='word2vec', 
        feature_size=100, 
        batch_size=500  # 根据内存情况调整批处理大小
    )
    
    # 创建文件生成器
    def combined_generator():
        yield from detector.load_files_generator(benign_dir, 0)
        yield from detector.load_files_generator(malicious_dir, 1)
    
    # 增量训练模型
    metrics = detector.fit_incremental(combined_generator())
    
    if metrics:
        print("\n最终评估结果:")
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")
    
    # 保存模型
    detector.save_model('webshell_detector_model.pkl')
    print("模型已保存到 webshell_detector_model.pkl")
    


if __name__ == "__main__":
    main()

