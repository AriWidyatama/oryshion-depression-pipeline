"""Modul Untuk Menentukan Parameter Terbaik Model"""

import json
import os
from typing import NamedTuple, Dict, Text, Any
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from kerastuner.engine import base_tuner
from kerastuner import Hyperband, Objective

from depression_transform import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name
)

# NamedTuple untuk Result
TunerResult = NamedTuple("TunerResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any]),
])

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_binary_accuracy",
    mode="max",
    verbose=1,
    patience=5,
)

def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")

def input_fn(file_pattern, tf_transform_output, batch_size=64):
    """Menghasilkan fitur dan label untuk tuning dan training."""
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset

# Model Building Function
def model_builder(hp):
    """
    Membangun model dengan hyperparameter tuning.

    Parameters:
    hp: Objek hyperparameter yang menentukan konfigurasi model.

    Returns:
    tf.keras.Model: Model Keras yang telah dikompilasi dengan arsitektur yang dapat dikonfigurasi.
    """

    input_features = []

    # Membuat input layer untuk fitur kategorikal
    for key, dim in CATEGORICAL_FEATURES.items():
        input_features.append(
            tf.keras.Input(shape=(dim + 1,), name=transformed_name(key))
        )

    # Membuat input layer untuk fitur numerikal
    for feature in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature))
        )

    # Menggabungkan input layer
    concatenate = tf.keras.layers.concatenate(input_features)

    # Hyperparameter tuning jumlah layer
    num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
    deep = concatenate

    # Hyperparameter tuning jumlah unit dan dopout
    for i in range(num_layers):
        units = hp.Int(f'units_{i}', min_value=16, max_value=256, step=16)
        deep = tf.keras.layers.Dense(units, activation='relu')(deep)
        dropout_rate = hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)
        if dropout_rate > 0:
            deep = tf.keras.layers.Dropout(dropout_rate)(deep)

    # Menggabungkan Hiden layer, output, dan input untuk Model
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(deep)
    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)

    # Hyperparameter tuning learning_rate
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    return model

# Tuner Function
def tuner_fn(fn_args: FnArgs):
    """
    Fungsi utama untuk melakukan hyperparameter tuning.
    
    Args:
        Objek yang berisi berbagai argumen yang diperlukan oleh Tuner dan Trainer.
    
    Returns:
        TunerResult: Hasil tuning yang berisi tuner dan parameter terbaik.
    """

    # Memuat hasil transformasi fitur
    transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Membaca data latih dan validasi
    train_data = input_fn(fn_args.train_files[0], transform_output)
    eval_data = input_fn(fn_args.eval_files[0], transform_output)

    # Inisialisasi tuner dengan algoritma Hyperband
    tuner = Hyperband(
        hypermodel=lambda hp: model_builder(hp),
        objective=Objective('val_binary_accuracy', direction='max'),
        max_epochs=5,
        factor=3,
        directory=fn_args.working_dir,
        project_name="depression_tuner",
    )

    # Melakukan pencarian hyperparameter terbaik
    tuner.search(
        x=train_data,
        validation_data=eval_data,
        steps_per_epoch=50,
        validation_steps=50,
        callbacks=[early_stopping],
        epochs=5
    )

    # Mengambil hyperparameter terbaik
    best_hps = tuner.get_best_hyperparameters(1)[0]
    best_hps_values = best_hps.values

    # Simpan hyperparameter ke file JSON
    with open(os.path.join('output', "best_hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(best_hps_values, f)

    # Log hyperparameter terbaik
    print(f"Best hyperparameters: {best_hps_values}")

    return TunerResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks": [early_stopping],
            "x": train_data,
            "validation_data": eval_data,
            "steps_per_epoch": 50,
            "validation_steps": 50,
        },
    )
