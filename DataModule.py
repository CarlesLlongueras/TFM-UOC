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
SAMPLE_ID = "sampleID"
DATA = "data.tsv"
LABELS = "labels.tsv"

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, non_survival=False):
        super().__init__()
        self.batch_size = batch_size
        self.non_survival = non_survival  # Si True, trabaja con la clase superviviente

        # Inicialización de variables para almacenar los datos
        self.survivor_features = self.survivor_labels = None
        self.non_survivor_features = self.non_survivor_labels = None
        self.train_data = self.val_data = self.test_data = None
        self.random_train_feature = self.random_test_feature = None
        self.random_train_labels = self.random_test_labels = None
        self.synthetic = self.real = None

        # Cargar el dataset existente o crear uno nuevo desde archivos TSV
        self.df = pq.read_table(MERGED).to_pandas() if os.path.exists(MERGED) else self.create_tsv()

        # Preparar los datos
        self.prepare_data()
        self.split_gan_data()  # Dividir datos para la GAN
        self.split_random_forest_data()  # Dividir datos para Random Forest

    def create_tsv(self):
        """Carga datos desde archivos TSV, los combina y guarda en formato Parquet"""
        # Procesar archivo de etiquetas
        labels_df = self.process_tsv(LABELS)
        labels_df = labels_df[[SAMPLE_ID, LABEL_COLUMN]]

        # Procesar archivo de características
        data_df = self.process_tsv(DATA, transpose=True)

        # Combinar datos y etiquetas usando "sampleID" como clave
        merged_df = data_df.merge(labels_df, on=SAMPLE_ID, how="inner")

        # Eliminar columna "sampleID" que no se usará en el modelo
        merged_df.drop(columns=[SAMPLE_ID], inplace=True)

        # Convertir todos los datos a numéricos y eliminar NaN
        merged_df = merged_df.dropna(axis=0)

        # Guardar el dataset combinado en formato Parquet
        pq.write_table(pa.Table.from_pandas(merged_df), MERGED)
        return merged_df

    def process_tsv(self, file_path, transpose=False):
        """Procesa un archivo TSV y lo convierte en DataFrame"""
        if transpose:
            # Transponer el dataframe y usar la primera columna como nombres de fila
            df = pd.read_csv(file_path, sep='\t')
            df = df.set_index(df.columns[0]).T.reset_index()
            df = df.rename(columns={"index": SAMPLE_ID})
        else:
            # Renombrar la primera columna a "sampleID"
            df = pd.read_csv(file_path, sep='\t')
            df = df.rename(columns={df.columns[0]: SAMPLE_ID})

        return df

    def prepare_data(self):
        """Divide los datos en supervivientes y no supervivientes"""
        if LABEL_COLUMN not in self.df.columns:
            raise ValueError(f"La columna de etiquetas '{LABEL_COLUMN}' no existe en el dataframe")

        # Datos de supervivientes (etiqueta 0)
        self.survivor_features, self.survivor_labels = self.split_features_labels(
            self.df[self.df[LABEL_COLUMN] == 0]
        )

        # Datos de no supervivientes (etiqueta 1)
        self.non_survivor_features, self.non_survivor_labels = self.split_features_labels(
            self.df[self.df[LABEL_COLUMN] == 1]
        )


    def split_features_labels(self, df):
        """Helper para separar características y etiquetas"""
        return df.drop(columns=[LABEL_COLUMN]), df[LABEL_COLUMN]

    def split_gan_data(self):
        """Prepara los datos para el entrenamiento de la GAN"""
        # Seleccionar características y etiquetas según la clase objetivo
        features = self.non_survivor_features if not self.non_survival else self.survivor_features
        labels = self.non_survivor_labels if not self.non_survival else self.survivor_labels

        # Convertir a tensores de PyTorch
        features_tensor = torch.tensor(features.values, dtype=torch.float32)
        labels_tensor = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1)

        # Crear dataset de PyTorch
        dataset = TensorDataset(features_tensor, labels_tensor)

        # Calcular tamaños para división entrenamiento/validación/prueba
        total = len(dataset)
        train_size = int(0.7 * total)  # 70% entrenamiento
        val_size = int(0.15 * total)  # 15% validación
        test_size = total - train_size - val_size  # 15% prueba

        # Dividir el dataset
        self.train_data, self.val_data, self.test_data = random_split(
            dataset, [train_size, val_size, test_size]
        )

    def add_synthetic(self, synthetic, all_synthetic=False):
        """Añade datos sintéticos al conjunto de entrenamiento"""

        # Cargar datos sintéticos generados por la GAN
        self.synthetic = synthetic

        if self.synthetic.empty:
            return

        # Asegurar que las columnas coincidan con los datos reales
        feature_cols = self.df.drop(columns=[LABEL_COLUMN]).columns
        self.synthetic.columns = feature_cols

        # Dividir datos reales (80% entrenamiento, 20% prueba)
        features = self.df.drop(columns=[LABEL_COLUMN])
        labels = self.df[LABEL_COLUMN]

        x_train_real, x_test_real, y_train_real, y_test_real = train_test_split(
            features, labels,
            test_size=0.2,
            random_state=42,
            stratify=labels  # Mantener proporción de clases
        )

        # Opción para reemplazar completamente los datos reales por sintéticos
        if all_synthetic:
            mask = y_train_real == (0 if self.non_survival else 1)
            x_train_real = x_train_real[~mask]
            y_train_real = y_train_real[~mask]

        # Crear etiquetas para los datos sintéticos
        synthetic_labels = pd.Series(
            [0 if self.non_survival else 1] * len(self.synthetic),
            name=LABEL_COLUMN
        )

        # Combinar datos reales y sintéticos SOLO en el conjunto de entrenamiento
        x_train_real = pd.concat([x_train_real, self.synthetic], ignore_index=True)
        y_train_real = pd.concat([y_train_real, synthetic_labels], ignore_index=True)

        self.random_train_feature = x_train_real
        self.random_train_labels = y_train_real
        self.random_test_feature = x_test_real
        self.random_test_labels = y_test_real

    def split_random_forest_data(self):
        """Divide los datos para el modelo Random Forest"""
        features = self.df.drop(columns=[LABEL_COLUMN])
        labels = self.df[LABEL_COLUMN]

        # División estratificada (80% entrenamiento, 20% prueba)
        (self.random_train_feature, self.random_test_feature,
         self.random_train_labels, self.random_test_labels) = train_test_split(
            features, labels,
            test_size=0.2,
            stratify=labels  # Mantener balance de clases
        )

    def train_dataloader(self):
        """DataLoader para el conjunto de entrenamiento"""
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size
        )

    def val_dataloader(self):
        """DataLoader para el conjunto de validación"""
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size
        )

    def test_dataloader(self):
        """DataLoader para el conjunto de prueba"""
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size
        )