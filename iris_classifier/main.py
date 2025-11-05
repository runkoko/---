from data_explorer import DataExplorer
from sklearn.datasets import load_iris
from data_explorer import DataExplorer
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from utils import save_model
import numpy as np


def main():
    # 加载数据
    iris = load_iris()

    # 数据探索
    explorer = DataExplorer(iris)
    df = explorer.explore()

    # 准备数据
    X, y = iris.data, iris.target

    # 训练模型
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.train(X, y, optimize=True)

    # 评估模型
    evaluator = ModelEvaluator(trainer.model, iris.target_names)
    accuracy = evaluator.evaluate(X_test, y_test)

    # 保存模型
    save_model(trainer.model)

    # 预测新样本示例
    new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = trainer.model.predict(new_sample)
    probability = trainer.model.predict_proba(new_sample)

    print(f"\n新样本预测结果: {iris.target_names[prediction][0]}")
    print("预测概率分布:")
    for i, prob in enumerate(probability[0]):
        print(f"  {iris.target_names[i]}: {prob:.4f}")


if __name__ == "__main__":
    main()