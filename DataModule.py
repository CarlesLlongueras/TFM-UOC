import os
import pytorch_lightning as pl
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.model_selection import train_test_split

# Definición de los archivos de datos
MERGED = "merged_dataset.parquet"
LABEL_COLUMN = "OS"
SYNTHETIC = "synthetic_samples.csv"

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

        # Inicialización de variables para los datasets
        self.survivor_features, self.survivor_labels = None, None
        self.non_survivor_features, self.non_survivor_labels = None, None
        self.train_data, self.val_data, self.test_data = None, None, None
        self.random_train_feature, self.random_test_feature = None, None
        self.random_train_labels, self.random_test_labels = None, None

        # Cargar el dataset si existe, de lo contrario, crearlo desde archivos TSV
        self.df = pq.read_table(MERGED).to_pandas() if os.path.exists(MERGED) else self.create_tsv()

        self.prepare_data()
        self.split_GAN_data()
        self.split_random_forest_data()

    def create_tsv(self):
        """Carga los datos desde archivos TSV, los combina y los guarda en formato Parquet."""
        labels_df = self.process_tsv("labels.tsv")
        data_df = self.process_tsv("data.tsv", transpose=True)

        # Unir datos y etiquetas usando "sampleID" como clave
        merged_df = data_df.merge(labels_df, on="sampleID", how="inner")

        # Eliminar la columna "sampleID" ya que no es necesaria para el modelo
        merged_df.drop(columns=["sampleID"], inplace=True)

        # Guardar el dataset combinado en formato Parquet
        pq.write_table(pa.Table.from_pandas(merged_df), MERGED)
        return merged_df

    def process_tsv(self, file_path, transpose=False):
        """Carga un archivo TSV y lo procesa para convertirlo en DataFrame."""
        df = pd.read_csv(file_path, sep='\t')

        if transpose:
            df = df.transpose()
            df.columns = df.iloc[0]  # La primera fila pasa a ser los nombres de las columnas
            df = df[1:].reset_index().rename(columns={"index": "sampleID"})  # Renombrar índice a "sampleID"
        else:
            df.rename(columns={df.columns[0]: "sampleID"}, inplace=True)  # Renombrar la primera columna
            sample_ids = df["sampleID"]
            df = df.select_dtypes(exclude=['object'])  # Filtrar solo datos numéricos
            df["sampleID"] = sample_ids  # Restaurar "sampleID"

        # Eliminar columnas con valores NaN
        return df.dropna(axis=1)

    def prepare_data(self):
        """Separa el dataset en supervivientes y no supervivientes."""
        survivors = self.df[self.df[LABEL_COLUMN] == 0]
        non_survivors = self.df[self.df[LABEL_COLUMN] == 1]

        self.survivor_features, self.survivor_labels = self.split_features_and_labels(survivors)
        self.non_survivor_features, self.non_survivor_labels = self.split_features_and_labels(non_survivors)

    def split_features_and_labels(self, dataframe):
        """Separa características y etiquetas del dataset."""
        features = dataframe.drop(columns=[LABEL_COLUMN])
        labels = dataframe[LABEL_COLUMN]
        return features, labels

    def split_GAN_data(self):
        """Divide los datos de no supervivientes en entrenamiento, validación y prueba para la GAN."""
        train_size = int(0.7 * len(self.non_survivor_features))
        val_size = int(0.15 * len(self.non_survivor_features))
        test_size = len(self.non_survivor_features) - train_size - val_size

        # Convertir datos a tensores
        features_tensor = torch.tensor(self.non_survivor_features.values, dtype=torch.float32)
        labels_tensor = torch.tensor(self.non_survivor_labels.values, dtype=torch.float32)

        # Crear dataset con características y etiquetas
        dataset = TensorDataset(features_tensor, labels_tensor)

        # División en conjuntos de entrenamiento, validación y prueba
        self.train_data, self.val_data, self.test_data = random_split(dataset, [train_size, val_size, test_size])

    def add_synthetic(self):
        """Añade datos sintéticos generados por la GAN al conjunto de datos original."""
        synthetic = pd.read_csv(SYNTHETIC)

        # Asegurar que los datos sintéticos tienen los mismos nombres de columnas
        features = self.df.drop(columns=[LABEL_COLUMN])
        synthetic.columns = features.columns

        # Concatenar los datos sintéticos con los datos reales
        features = pd.concat([features, synthetic], ignore_index=True)

        # Crear etiquetas para los datos sintéticos (suponiendo que son no supervivientes)
        new_labels = pd.DataFrame({0: [1] * len(synthetic)})

        # Combinar etiquetas originales con las nuevas etiquetas sintéticas
        labels = pd.concat([self.df[LABEL_COLUMN], new_labels], ignore_index=True)

        # Dividir en conjunto de entrenamiento y prueba
        (self.random_train_feature, self.random_test_feature,
         self.random_train_labels, self.random_test_labels) = train_test_split(features, labels, test_size=0.2)

    def split_random_forest_data(self):
        """Divide los datos originales en entrenamiento y prueba para Random Forest."""
        features = self.df.drop(columns=[LABEL_COLUMN])
        labels = self.df[LABEL_COLUMN]

        (self.random_train_feature, self.random_test_feature,
         self.random_train_labels, self.random_test_labels) = train_test_split(features, labels, test_size=0.2)

    def train_dataloader(self):
        """Devuelve el DataLoader para el conjunto de entrenamiento."""
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """Devuelve el DataLoader para el conjunto de validación."""
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        """Devuelve el DataLoader para el conjunto de prueba."""
        return DataLoader(self.test_data, batch_size=self.batch_size)