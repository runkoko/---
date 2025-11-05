import joblib
import datetime


def save_model(model, filename=None):
    """保存模型到文件"""
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'iris_classifier_{timestamp}.pkl'

    joblib.dump(model, filename)
    print(f"模型已保存为: {filename}")
    return filename


def load_model(filename):
    """从文件加载模型"""
    return joblib.load(filename)