import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np


class ModelEvaluator:
    def __init__(self, model, target_names):
        self.model = model
        self.target_names = target_names

    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"测试集准确率: {accuracy:.4f}")
        print("\n详细分类报告:")
        print(classification_report(y_test, y_pred, target_names=self.target_names))

        self._plot_confusion_matrix(y_test, y_pred)

        # 交叉验证
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5)
        print(f"交叉验证分数: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")

        return accuracy

    def _plot_confusion_matrix(self, y_test, y_pred):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.target_names, yticklabels=self.target_names)
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.show()