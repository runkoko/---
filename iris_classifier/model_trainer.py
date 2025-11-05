from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.best_params_ = None
        self.best_score_ = None

    def train(self, X, y, optimize=True, test_size=0.2):
        """训练模型"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        if optimize:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier())
            ])

            param_grid = {
                'knn__n_neighbors': range(1, 15),
                'knn__weights': ['uniform', 'distance'],
                'knn__metric': ['euclidean', 'manhattan']
            }

            self.model = GridSearchCV(
                pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
        else:
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier(n_neighbors=3))
            ])

        self.model.fit(X_train, y_train)

        if optimize:
            self.best_params_ = self.model.best_params_
            self.best_score_ = self.model.best_score_
            print(f"最佳参数: {self.best_params_}")
            print(f"最佳交叉验证分数: {self.best_score_:.4f}")

        return X_train, X_test, y_train, y_test