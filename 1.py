# 第一步：导入必要的库和数据加载
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 查看数据基本信息
print("数据集形状:", X.shape)
print("类别数量:", len(np.unique(y)))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 第二步：创建模型实例
model = KNeighborsClassifier(n_neighbors=3)

# 第三步：训练模型
model.fit(X_train, y_train)

# 第四步：预测
y_pred = model.predict(X_test)

# 第五步：评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.2f}")

# 详细分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 预测新样本示例
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # 新的鸢尾花测量数据
prediction = model.predict(new_sample)
print(f"\n新样本预测结果: {iris.target_names[prediction][0]}")
# 查看特征名称
print("特征名称:")
for i, feature_name in enumerate(iris.feature_names):
    print(f"  {i+1}. {feature_name}")

# 查看特征值的统计信息
print("\n特征值统计信息:")
print(f"特征值形状: {iris.data.shape}")
print("各特征的范围:")
for i, feature_name in enumerate(iris.feature_names):
    feature_data = iris.data[:, i]
    print(f"  {feature_name}: {feature_data.min():.1f} - {feature_data.max():.1f}")

# 显示前5个样本的特征值
print("\n前5个样本的特征值:")
print("花萼长度 | 花萼宽度 | 花瓣长度 | 花瓣宽度")
print("-" * 35)
for i in range(5):
    print(f"{iris.data[i][0]:8.1f} | {iris.data[i][1]:8.1f} | {iris.data[i][2]:8.1f} | {iris.data[i][3]:8.1f}")

