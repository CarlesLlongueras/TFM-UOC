import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import umap.umap_ as umap
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.manifold import TSNE, MDS

from DataModule import DataModule
from DimensionalityMethods import DimensionalityMethods
from GAN import GAN
from Inform import Inform

LABEL_COLUMN = "OS"  # Nombre de la columna que contiene las etiquetas

class Dimensionality:

    def get_result(self, method):
        results = []
        data_module = DataModule(batch_size=32)
        or_lab_df = data_module.df[LABEL_COLUMN]  # Etiquetas originales
        or_fea_df = data_module.df.drop(columns=[LABEL_COLUMN])  # Features originales
        comp2 = []
        inform = None
        list_range = range(2, 3, 2)
        for n_components in list_range:
            # Aplica reducción de dimensionalidad
            reduced_data = self.apply_method(or_lab_df, or_fea_df, n_components, method)
            or_lab_df = pd.DataFrame(or_lab_df.values, columns=[LABEL_COLUMN])
            data_module.df = pd.concat([pd.DataFrame(reduced_data), or_lab_df], axis=1)

            if n_components == 2:
                comp2 = pd.DataFrame(reduced_data)  # Se guarda para graficar

            # Prepara y divide los datos
            data_module.prepare_data()
            data_module.split_gan_data()
            data_module.split_random_forest_data()

            # Entrena la GAN con los datos reducidos
            output_dim = reduced_data.shape[1]
            model = GAN(output_dim=output_dim, end_print=False, lr=0.00015)
            trainer = pl.Trainer(max_epochs=100, accelerator="auto")
            trainer.fit(model, datamodule=data_module)

            # Obtiene estadísticas intermedias del modelo
            inform = Inform(data_module, model)
            statistics = inform.get_mid_statistics()[1]
            results.append([n_components, statistics])

        # Grafica los componentes 2D si se usaron
        self.plot_2d_components(comp2, or_lab_df)
        # Imprime el informe de resultados
        inform.print_report(results, list_range)
        return results

    def apply_method(self, labels, data_set, n_components, method):
        if method == DimensionalityMethods.PCA:
            return self.pca(data_set, n_components)
        elif method == DimensionalityMethods.UMAP:
            return self.umap(data_set, n_components)
        elif method == DimensionalityMethods.MDS:
            return self.multi_scaling(data_set, n_components)
        elif method == DimensionalityMethods.TSNE:
            return self.tsne(data_set, n_components)
        elif method == DimensionalityMethods.LASSO:
            return self.lasso(labels, data_set, n_components)
        else:
            raise Exception("Método no soportado")

    # Métodos individuales de reducción de dimensionalidad
    def pca(self, data_set, n_components):
        return PCA(n_components=n_components).fit_transform(data_set)

    def umap(self, data_set, n_components):
        return umap.UMAP(n_components=n_components).fit_transform(data_set)

    def multi_scaling(self, data_set, n_components):
        return MDS(n_components=n_components, random_state=42).fit_transform(data_set)

    def tsne(self, data_set, n_components):
        return TSNE(n_components=n_components, method='exact', random_state=42).fit_transform(data_set)

    def lasso(self, labels, data_set, n_components):
        model = Lasso(alpha=0.01, max_iter=1000)
        model.fit(data_set, labels)
        idx = np.argsort(np.abs(model.coef_))[-n_components:]  # Selección de las variables más relevantes
        return data_set.iloc[:, idx]

    # Gráfica de evolución de rendimiento según número de componentes
    def plot_evolution(self, results):
        df = pd.DataFrame(results, columns=['n_components', 'accuracy'])
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='n_components', y='accuracy', marker='o', linewidth=2.2)
        plt.title('Evolución del rendimiento')
        plt.xlabel('Número de Componentes')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Gráfico 2D para visualizar separación por clase
    def plot_2d_components(self, components, y):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=components.iloc[:, 0], y=components.iloc[:, 1], hue=y.iloc[:, 0], palette="Set1", s=60, alpha=0.8)
        plt.title("Separación por grupos")
        plt.xlabel(components.columns[0])
        plt.ylabel(components.columns[1])
        plt.legend(title="Clase")
        plt.tight_layout()
        plt.show()
