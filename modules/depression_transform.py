"""Modul Transform Untuk Prepocessing Teks."""

import tensorflow as tf
import tensorflow_transform as tft

# Pembagian jenis Fitur
CATEGORICAL_FEATURES = {
    "Dietary Habits": 4,  # Kebiasaan Makan
    "Family History of Mental Illness": 2,  # Riwayat Keluarga dengan Gangguan Mental
    "Have you ever had suicidal thoughts ?": 2,  # Pernahkah Anda Memiliki Pikiran Bunuh Diri?
    "Sleep Duration": 5,  # Durasi Tidur
}

NUMERICAL_FEATURES = [
    "Academic Pressure",  # Tekanan Akademik
    "CGPA",  # IPK
    "Financial Stress",  # Stres Keuangan
    "Study Satisfaction",  # Kepuasan Studi
    "Work/Study Hours",  # Jam Kerja/Belajar
]

LABEL_KEY = "Depression"  # Label Terger Depresi atau tidak

def transformed_name(key: str) -> str:
    """Mengubah nama fitur yang di transform"""
    return f"{key}_xf"

def convert_num_to_one_hot(label_tensor: tf.Tensor, num_labels: int = 2) -> tf.Tensor:
    """
    Mengonversi label numerik menjadi one-hot.
    Args:
        int: label_tensor (0 or 1)
    Returns
        label tensor
    """
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])

def preprocessing_fn(inputs: dict) -> dict:
    """
    Melakukan preprocessing fitur input menjadi transform fitur.
    Args:
        inputs: fitur dalam bentuk raw, baik berupa integer maupun teks.
    Return:
        outputs: fitur transform.
    """
    outputs = {}

    # Preprocessing fitur kategorikal
    for key, dim in CATEGORICAL_FEATURES.items():
        int_value = tft.compute_and_apply_vocabulary(inputs[key], top_k=dim + 1)
        outputs[transformed_name(key)] = convert_num_to_one_hot(int_value, num_labels=dim + 1)

    # Preprocessing fitur numerik
    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])

    # Preprocessing label
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
