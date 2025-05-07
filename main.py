import pytorch_lightning as pl
from DataModule import DataModule
from Dimensionality import Dimensionality
from DimensionalityMethods import DimensionalityMethods
from Inform import Inform
from GAN import GAN

# Parámetros del modelo
INPUT_DIM = 100  # Dimensión del vector de ruido para el generador
BATCH_SIZE = 64  # Tamaño del lote
LR = 0.00014  # Tasa de aprendizaje

def global_statistics():
    data_module = DataModule(batch_size=BATCH_SIZE)
    output_dim = data_module.df.shape[1]-1
    model = GAN(output_dim, INPUT_DIM, LR, True)
    trainer = pl.Trainer(
        max_epochs=440,  # Número máximo de épocas de entrenamiento
        accelerator="auto",  # Utiliza GPU si está disponible, de lo contrario CPU
    )
    trainer.fit(model, datamodule=data_module)
    inform = Inform(data_module, model)
    inform.fill_data()
    inform.get_statistics()

def gan_statistics():
    data_module = DataModule(batch_size=BATCH_SIZE)
    inform = Inform(data_module, None)
    inform.gan_statistics(False, False, True)

def dimensionality():
    dim = Dimensionality()
    dim.get_result(DimensionalityMethods.LASSO)

if __name__ == '__main__':
    global_statistics()
    #gan_statistics()
    #dimensionality()