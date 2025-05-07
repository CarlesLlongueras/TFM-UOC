import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import numpy as np
from GAN import GAN
from Predictor import Predictor
import matplotlib.pyplot as plt


class Inform:
    def __init__(self, data_module, gan):
        # Porcentajes de generación de muestras sintéticas
        self.perc = [0, 0.1, 0.25, 0.5, 0.75, 1]
        # Etiquetas descriptivas para los porcentajes
        self.perc_Names = ["Zero", "Bajo", "Medio bajo", "Medio", "Medio alto", "Alto", "Todo"]
        self.data_Module = data_module
        self.gan = gan
        # Número de muestras por clase
        self.n_Major = len(data_module.survivor_features)
        self.n_Minor = len(data_module.non_survivor_features)
        # Almacena datasets por porcentaje y estadísticas correspondientes
        self.data_perc = []
        self.statistics_perc = []
        self.synthetic = []

    def fill_data(self):
        # Cálculo de la diferencia de tamaño entre clases
        diff = self.n_Major - self.n_Minor
        for p in self.perc:
            # Generación de muestras sintéticas según el porcentaje
            self.gan.generate_samples(self.data_Module, int(diff * p))
            # Almacena conjunto de datos sintetizados para entrenamiento/prueba
            it = [self.data_Module.random_train_feature, self.data_Module.random_test_feature,
                  self.data_Module.random_train_labels, self.data_Module.random_test_labels]
            self.synthetic.append(self.data_Module.synthetic)
            self.data_perc.append(it)

        # Agrega el caso de "generar All"
        self.all_case()

    def all_case(self):
        # Generación de tantas muestras como la clase mayoritaria (igualar clases)
        self.gan.generate_samples(self.data_Module, self.n_Major, True)
        it = [self.data_Module.random_train_feature, self.data_Module.random_test_feature,
              self.data_Module.random_train_labels, self.data_Module.random_test_labels]
        self.synthetic.append(self.data_Module.synthetic)
        self.data_perc.append(it)

    def gan_statistics(self, lr=False, batch=False, epochs=False):
        # Listas de valores a testear para cada parámetro
        lr_list = np.arange(0.00010, 0.00050, 0.00004)
        batch_list = np.arange(16, 128, 16)
        epochs_list = np.arange(300, 500, 20)

        if lr:
            lr_statistics = []
            for lr_value in lr_list:
                # Entrena GAN con distintos learning rates
                lr_statistics.append(self.init_statistics_gan(lr=float(lr_value)))
            self.show_gan_plots("Learning Rate", lr_list, lr_statistics)

        if batch:
            batch_statistics = []
            original_batch = self.data_Module.batch_size
            for batch_value in batch_list:
                # Entrena GAN con distintos batch sizes
                batch_statistics.append(self.init_statistics_gan(batch=int(batch_value)))
            self.show_gan_plots("Batch", batch_list, batch_statistics)
            # Restaura batch original
            self.data_Module.batch_size = original_batch

        if epochs:
            epochs_statistics = []
            for epochs_value in epochs_list:
                # Entrena GAN con distintas épocas
                epochs_statistics.append(self.init_statistics_gan(epochs=int(epochs_value)))
            self.show_gan_plots("Epochs", epochs_list, epochs_statistics)

    def show_gan_plots(self, label, metric_list, statistics):
        # Gráfico de línea para ver cómo cambia la precisión según el parámetro
        plt.figure(figsize=(8, 5))
        plt.plot(metric_list, statistics, marker='o', linestyle='-')
        plt.title(f"Estadísticas del modelo según {label}")
        plt.xlabel(label)
        plt.ylabel("Precision")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def init_statistics_gan(self, lr=0.00030, batch=32, epochs=50):
        # Inicializa y entrena un nuevo GAN con los parámetros dados
        output_dim = self.data_Module.df.shape[1] - 1
        self.data_Module.original_batch = batch
        model = GAN(output_dim=output_dim, lr=lr, end_print=False)
        trainer = pl.Trainer(max_epochs=int(epochs), accelerator="auto")
        trainer.fit(model, datamodule=self.data_Module)
        # Devuelve puntuación con predictor tras generar muestras
        return self.get_mid_statistics(model)[0]

    def get_mid_statistics(self, gan=None):
        # Usa el GAN entrenado para generar muestras y evaluar con un predictor
        if gan is None:
            gan = self.gan
        diff = (self.n_Major - self.n_Minor)
        gan.generate_samples(self.data_Module, int(diff * 0.4))
        predictor = Predictor()
        predictor.fit_model(self.data_Module.random_train_feature, self.data_Module.random_train_labels)
        predictor.predict(self.data_Module.random_test_feature)
        return [predictor.get_score(self.data_Module.random_test_labels), predictor.get_report()]

    def get_statistics(self):
        # Entrena y evalúa un predictor con todos los subconjuntos generados
        predictor = Predictor()
        for train_F, test_F, train_L, test_L in self.data_perc:
            predictor.fit_model(train_F, train_L)
            predictor.predict(test_F)
            it = [predictor.get_score(test_L), predictor.get_report(),
                  predictor.get_features_importance(train_F.columns)]
            self.statistics_perc.append(it)
        self.print_score()
        self.print_report()
        self.print_features_importance()

    def print_features_importance(self):
        # Gráfico tipo heatmap con las 10 características más importantes por iteración
        features_list = [stat[2] for stat in self.statistics_perc]
        renamed_dfs = []

        for i, df in enumerate(features_list):
            df_iter = df.copy().reset_index()
            df_iter.columns = ["Feature", f"Importancia_{self.perc_Names[i]}"]
            df_iter = df_iter.set_index("Feature")
            df_iter = df_iter.sort_values(by=f"Importancia_{self.perc_Names[i]}", ascending=False).head(10)
            renamed_dfs.append(df_iter)

        df_combined = pd.concat(renamed_dfs, axis=1)
        plt.figure(figsize=(16, 10))
        sns.heatmap(df_combined, cmap="coolwarm", linewidths=0.5, annot=True)
        plt.title("Importancia de las Características por Iteración")
        plt.xlabel("Iteración")
        plt.ylabel("Características")
        plt.tight_layout()
        plt.show()

    def print_report(self, statistics_perc=None, perc_names=None):
        # Imprime evolución de métricas de clasificación por clase y porcentaje
        if statistics_perc is None:
            statistics_perc = self.statistics_perc
        if perc_names is None:
            perc_names = self.perc_Names
        reports_list = [stat[1] for stat in statistics_perc]
        filtered_rows = []
        for report in reports_list:
            for row in report.values:
                filtered_rows.append(row)

        df_report = pd.DataFrame(filtered_rows, columns=["Clase", "Precision", "Recall", "F1-score"])
        df_report["Iteración"] = [perc_names[i // 2] for i in
                                  range(len(df_report))]
        df_long = pd.melt(df_report, id_vars=["Clase", "Iteración"], var_name="Métrica", value_name="Valor")
        plt.figure(figsize=(10, 6))
        sns.lineplot(x="Iteración", y="Valor", hue="Métrica", style="Clase", markers=True, data=df_long)
        plt.title("Evolución de Métricas por Iteración y Clase")
        plt.xlabel("Iteración")
        plt.ylabel("Valor")
        plt.legend(title="Métricas")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def print_score(self):
        # Gráfico de barras con la puntuación global por porcentaje de muestras sintéticas
        scores = [stat[0] for stat in self.statistics_perc]
        plt.figure(figsize=(6, 4))
        plt.bar(self.perc_Names, scores, color='teal', width=0.4)
        plt.ylim(min(scores) - 0.01, max(scores) + 0.01)
        plt.title("Score del modelo predictor")
        plt.ylabel("Score")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
