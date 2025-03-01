from typing import NamedTuple, Dict, Text, Any
from tfx.components.trainer.fn_args_utils import FnArgs
from kerastuner.engine import base_tuner
from kerastuner import Hyperband, Objective
import tensorflow as tf
import tensorflow_transform as tft
import json
import os

from depression_transform import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name
)

TunerResult = NamedTuple("TunerResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any]),
])

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
    """Generates features and labels for tuning/training."""
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

def model_builder(hp):
    input_features = []

    for key, dim in CATEGORICAL_FEATURES.items():
        input_features.append(
            tf.keras.Input(shape=(dim + 1,), name=transformed_name(key))
        )
    
    for feature in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature))
        )
    
    concatenate = tf.keras.layers.concatenate(input_features)
    
    num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
    deep = concatenate
    
    for i in range(num_layers):
        units = hp.Int(f'units_{i}', min_value=16, max_value=256, step=16)
        deep = tf.keras.layers.Dense(units, activation='relu')(deep)
        dropout_rate = hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)
        if dropout_rate > 0:
            deep = tf.keras.layers.Dropout(dropout_rate)(deep)
    
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(deep)
    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)
    
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    
    return model

def tuner_fn(fn_args: FnArgs):
    transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_data = input_fn(fn_args.train_files[0], transform_output)
    eval_data = input_fn(fn_args.eval_files[0], transform_output)

    tuner = Hyperband(
        hypermodel=lambda hp: model_builder(hp),
        objective=Objective('val_binary_accuracy', direction='max'),
        max_epochs=5,
        factor=3,
        directory=fn_args.working_dir,
        project_name="depression_tuner",
    )

    tuner.search(
        x=train_data,
        validation_data=eval_data,
        steps_per_epoch=50,
        validation_steps=50,
        callbacks=[early_stopping],
        epochs=5
    )
    
    best_hps = tuner.get_best_hyperparameters(1)[0]
    best_hps_values = best_hps.values
    
    with open(os.path.join('output', "best_hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(best_hps_values, f)
    
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
