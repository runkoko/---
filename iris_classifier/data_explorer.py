import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class DataExplorer:
    def __init__(self, iris_data):
        self.iris = iris_data
        self.df = None

    def explore(self):
        """探索数据并生成基本信息和可视化"""
        X, y = self.iris.data, self.iris.target
        self.df = pd.DataFrame(X, columns=self.iris.feature_names)
        self.df['target'] = y
        self.df['target_name'] = [self.iris.target_names[i] for i in y]

        print("数据基本信息:")
        print(self.df.describe())

        print("\n类别分布:")
        print(self.df['target_name'].value_counts())

        self._plot_distributions()
        self._plot_correlation_heatmap()

        return self.df

    def _plot_distributions(self):
        """绘制特征分布直方图"""
        feature_names = self.iris.feature_names
        target_names = self.iris.target_names

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for i, feature in enumerate(feature_names):
            row, col = i // 2, i % 2
            for target in np.unique(self.df['target']):
                axes[row, col].hist(self.df[self.df['target'] == target][feature],
                                    alpha=0.7, label=target_names[target])
            axes[row, col].set_title(f'{feature}分布')
            axes[row, col].legend()
        plt.tight_layout()
        plt.show()

    def _plot_correlation_heatmap(self):
        """绘制特征相关性热力图"""
        plt.figure(figsize=(8, 6))
        corr = self.df[self.iris.feature_names].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('特征相关性热力图')
        plt.show()