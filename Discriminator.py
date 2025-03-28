import torch.nn as nn

class Discriminator(nn.Module):
    """
    Discriminador de la GAN encargado de clasificar si una muestra es real o generada.

    Parámetros:
        - input_dim: Dimensión de la entrada (igual a la salida del generador).
    """

    def __init__(self, input_dim):
        super().__init__()

        self.disc = nn.Sequential(
            nn.Linear(input_dim, 256), # Capa de entrada que reduce la dimensión a 256 neuronas
            nn.LeakyReLU(0.2),  # Activación con pendiente negativa de 0.2 para evitar gradientes muertos

            nn.Linear(256, 128), # Capa oculta intermedia con 128 neuronas
            nn.LeakyReLU(0.2),

            nn.Linear(128, 1), # Capa de salida con 1 neurona
            nn.Sigmoid() # Convierte la salida en una probabilidad entre 0 y 1 (real o falso)
        )

    def forward(self, x):
        return self.disc(x)