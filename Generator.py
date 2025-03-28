import torch.nn as nn

class Generator(nn.Module):
    """
    Generador para una Red Adversaria Generativa (GAN).
    Toma como entrada un vector aleatorio (ruido) y genera datos sintéticos.

    Parámetros:
        - input_dim (int): Dimensión del vector de entrada (ruido latente).
        - output_dim (int): Dimensión de la salida generada (misma que los datos reales).
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(input_dim, 128), # Capa de entrada: transforma la dimensión de entrada a 128 neuronas
            nn.LeakyReLU(0.2), # Función de activación con pendiente negativa de 0.2 para evitar neuronas muertas
            nn.BatchNorm1d(128),  #Normalización para estabilizar el entrenamiento y mejorar la convergencia

            nn.Linear(128, 512), # Capa oculta intermedia con 512 neuronas
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),

            nn.Linear(512, output_dim), # Capa de salida ajusta la dimensión al tamaño de los datos reales
        )

    def forward(self, x):
        return self.gen(x)