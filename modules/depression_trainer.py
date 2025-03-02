"""Modul Untuk Membangun dan Melatih Model"""

import os
import json
from tfx.components.trainer.fn_args_utils import FnArgs
import tensorflow as tf
import tensorflow_transform as tft

from depression_transform import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name
)
from depression_tuner import input_fn

def get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )

        transformed_features = model.tft_layer(parsed_features)

        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn

def build_model(hparams):
    """
    Membangun model Keras berdasarkan hyperparameter terbaik.

    Parameters:
    hparams: Hyperparameter yang digunakan untuk membangun model.

    Returns:
    tf.keras.Model: Model yang telah dikompilasi.
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
    deep = concatenate

    # Menggunakan hyperparameter yang telah ditemukan
    for i in range(hparams['num_layers']):
        deep = tf.keras.layers.Dense(hparams[f'units_{i}'], activation='relu')(deep)
        dropout_rate = hparams.get(f'dropout_{i}', 0.0)
        if dropout_rate > 0:
            deep = tf.keras.layers.Dropout(dropout_rate)(deep)

    # Menggabungkan Hiden layer, output, dan input untuk Model
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(deep)
    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)

    # Kompilasi Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate']),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    return model

def run_fn(fn_args: FnArgs):
    """Fungsi utama untuk Trainer di TFX."""

    # Load hasil transformasi
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # Membaca data latih dan validasi
    train_dataset = input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output)

    # Mengambil hasil hyperparameters terbaik
    best_hps = os.path.join('output', 'best_hyperparameters.json')
    with open(best_hps, "r", encoding="utf-8") as f:
        best_hparams = json.load(f)

    # Membangun model dengan hyperparameters terbaik
    model = build_model(best_hparams)
    model.summary()

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )

    # Konfigurasi training
    model.fit(
        x=train_dataset,
        epochs=20,
        validation_data=eval_dataset,
        steps_per_epoch=50,
        validation_steps=25,
        callbacks=[tensorboard_callback],
    )

    # Menyimpan hasil training
    signatures = {
        "serving_default": get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }

    model.save(
        fn_args.serving_model_dir, save_format="tf", signatures=signatures
    )
