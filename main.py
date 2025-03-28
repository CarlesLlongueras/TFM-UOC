import pytorch_lightning as pl

from DataModule import DataModule
from Predictor import Predictor
from GAN import GAN

# Parámetros del modelo
INPUT_DIM = 100  # Dimensión del vector de ruido para el generador
BATCH_SIZE = 32  # Tamaño del lote
LR = 0.0003  # Tasa de aprendizaje

if __name__ == '__main__':
    # Cargar el módulo de datos
    data_module = DataModule(batch_size=BATCH_SIZE)

    # Inicializar la GAN
    output_dim = data_module.df.shape[1]-1
    model = GAN(INPUT_DIM, output_dim, BATCH_SIZE, LR)

    # Configurar el entrenador
    trainer = pl.Trainer(
        max_epochs=50,  # Número máximo de épocas de entrenamiento
        accelerator="auto"  # Utiliza GPU si está disponible, de lo contrario CPU
    )

    # Entrenar la GAN
    trainer.fit(model, datamodule=data_module)

    # Generar muestras sintéticas después del entrenamiento
    model.generate_samples(num_samples=100)

    # Integrar muestras sintéticas en el dataset
    data_module.add_synthetic()

    # Entrenar y evaluar un modelo de predicción
    predictor = Predictor()
    predictor.fit_model(data_module.random_train_feature, data_module.random_train_labels)
    predictor.predict(data_module.random_test_feature)
    predictor.get_score(data_module.random_test_labels)
    predictor.get_report()
    predictor.get_features_importance(data_module.random_test_feature.columns, sorted=True)
