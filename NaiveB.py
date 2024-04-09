import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report
from joblib import dump, load


# 加载数据
data_path = 'fake_or_real_news.csv'
data = pd.read_csv(data_path)

# 数据预处理
# 合并标题和正文作为特征，将标签转换成二进制变量（1代表真新闻，0代表假新闻）
data['full_text'] = data['title'] + " " + data['text']
data['label_bin'] = data['label'].map({'REAL': 1, 'FAKE': 0})

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(data['full_text'], data['label_bin'], test_size=0.2, random_state=42)

# 创建模型管道，包含TF-IDF向量化和多项式朴素贝叶斯分类器
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# 训练模型
model.fit(X_train, y_train)

# 保存模型
model_path = 'fake_news_classifier.joblib'
dump(model, model_path)


# 模型评估
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)


# 输入待分类的新闻文本
full_text = data['full_text'][0]

# 使用模型预测文本的分类概率
predicted_probabilities = model.predict_proba([full_text])

# 输出为真新闻的概率
probability_of_real = predicted_probabilities[0][1]
print(f"Probability of being real news: {probability_of_real:.2%}")
