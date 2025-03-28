import pandas as pd
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class Predictor:
    def __init__(self):
        # Inicializa el modelo Random Forest y variables para predicción y evaluación
        self.rf = RandomForestClassifier()
        self.y_prediction = None
        self.X_test = None
        self.y_test = None

    def fit_model(self, x_train, y_train):
        """Entrena el modelo con datos de entrada convertidos a formato NumPy."""
        X_train = torch.tensor(x_train.values, dtype=torch.float32).numpy()
        self.rf.fit(X_train, y_train)

    def predict(self, x_test):
        """Genera predicciones a partir de datos de prueba."""
        self.X_test = torch.tensor(x_test.values, dtype=torch.float32).numpy()
        self.y_prediction = self.rf.predict(self.X_test)

    def get_score(self, y_test):
        """Devuelve la precisión del modelo."""
        if self.y_prediction is not None:
            self.y_test = y_test
            return self.rf.score(self.X_test, y_test)

    def get_report(self):
        """Muestra un informe de métricas si se realizaron predicciones."""
        if self.y_test is not None and self.y_prediction is not None:
            print(classification_report(self.y_test, self.y_prediction))
        else:
            print("No hay la información necesaria para generar el informe")

    def get_features_importance(self, columns, sorted=True):
        """Imprime la importancia de las características, con opción de ordenarlas."""
        features = pd.DataFrame(self.rf.feature_importances_, index=columns, columns=['Importancia'])
        if sorted:
            features = features.sort_values(by='Importancia', ascending=False)
        print(features)

    def define_hyperparameters(self, n_estimators=1000, min_samples_split=10, max_depth=20, random_state=32):
        """Define hiperparámetros personalizados para el modelo Random Forest."""
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            random_state=random_state
        )