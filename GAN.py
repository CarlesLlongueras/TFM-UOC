import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import torch

from Discriminator import Discriminator
from Generator import Generator


# Función para calcular la pérdida adversarial usando Binary Cross Entropy
def adversarial_loss(y_hat, y):
    return F.binary_cross_entropy_with_logits(y_hat, y)


# Inicialización de pesos usando Xavier
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)


class GAN(pl.LightningModule):
    def __init__(self, input_dim, output_dim, batch_size, lr):
        super().__init__()
        self.automatic_optimization = False  # Control manual de optimización
        self.save_hyperparameters()
        self.lr = lr  # Tasa de aprendizaje
        self.generator = Generator(input_dim, output_dim)
        self.discriminator = Discriminator(output_dim)

        # Aplicar inicialización de pesos a las redes
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        # Almacenar pérdidas para visualización
        self.train_g_loss = []
        self.train_d_loss = []

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch):
        data, _ = batch  # Solo se usan los datos, no las etiquetas
        opt_g, opt_d = self.optimizers()

        batch_size = data.shape[0]
        z = torch.randn(batch_size, self.hparams.input_dim, device=self.device)  # Ruido aleatorio

        # Etiquetas reales y falsas
        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        # ---- Entrenamiento del Generador ----
        self.toggle_optimizer(opt_g)
        g_loss = adversarial_loss(self.discriminator(self(z)), valid)
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        # ---- Entrenamiento del Discriminador ----
        self.toggle_optimizer(opt_d)
        real_loss = adversarial_loss(self.discriminator(data), valid)
        fake_loss = adversarial_loss(self.discriminator(self(z).detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        # Registrar pérdidas y precisión del discriminador
        self.log_dict({"d_loss": d_loss, "g_loss": g_loss})
        self.train_g_loss.append(g_loss.item())
        self.train_d_loss.append(d_loss.item())

        # Calcular precisión del discriminador
        with torch.no_grad():
            synthetic_data = self.generator(z).detach()
            discriminator_acc = self.calculate_discriminator_accuracy(data, synthetic_data)
            self.log("discriminator_accuracy", discriminator_acc, prog_bar=True)

    def calculate_discriminator_accuracy(self, real_data, fake_data):
        batch_size = real_data.size(0)

        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        real_acc = ((torch.sigmoid(self.discriminator(real_data)) > 0.5) == valid).float().mean()
        fake_acc = ((torch.sigmoid(self.discriminator(fake_data)) <= 0.5) == fake).float().mean()

        return ((real_acc + fake_acc) / 2).item()

    def validation_step(self, batch):
        data, _ = batch
        batch_size = data.size(0)
        z = torch.randn(batch_size, self.hparams.input_dim, device=self.device)
        synthetic_data = self.generator(z)

        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        real_loss = adversarial_loss(self.discriminator(data), valid)
        fake_loss = adversarial_loss(self.discriminator(synthetic_data.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        self.log("val_d_loss", d_loss, prog_bar=True)
        return d_loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return [opt_g, opt_d], []

    def on_train_end(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_g_loss, label="Generador (G) Loss", color='blue', linewidth=2)
        plt.plot(self.train_d_loss, label="Discriminador (D) Loss", color='orange', linewidth=2)
        plt.xlabel("Épocas")
        plt.ylabel("Pérdida")
        plt.title("Evolución de la Pérdida durante el Entrenamiento")
        plt.legend()
        plt.grid(True)
        plt.show()

    def generate_samples(self, num_samples=20):
        self.generator.eval()
        z = torch.randn(num_samples, self.hparams.input_dim, device=self.device)
        with torch.no_grad():
            synthetic_samples = self.generator(z)
        pd.DataFrame(synthetic_samples.cpu().numpy()).to_csv("synthetic_samples.csv", index=False)