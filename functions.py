import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
import tensorflow.keras.backend as K

from pathlib import Path
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

#from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

#from collections import Counter

tf.config.run_functions_eagerly(True)

# Weighted F1 Score for tuning and training
def compute_class_weights(y_train, num_classes):
    # Compute class weights for unbalanced datasets
    y_train_flat = np.argmax(y_train, axis=1) if num_classes > 2 else y_train.flatten()
    class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_train_flat)
    return dict(enumerate(class_weights))

class WeightedF1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="weighted_f1_score", **kwargs):
        super(WeightedF1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(shape=(num_classes,), initializer="zeros")
        self.false_positives = self.add_weight(shape=(num_classes,), initializer="zeros")
        self.false_negatives = self.add_weight(shape=(num_classes,), initializer="zeros")
        self.support = self.add_weight(shape=(num_classes,), initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1) if self.num_classes > 2 else tf.round(y_pred)
        y_true = tf.argmax(y_true, axis=-1) if self.num_classes > 2 else tf.squeeze(y_true)

        for i in range(self.num_classes):
            y_true_i = tf.cast(y_true == i, tf.float32)
            y_pred_i = tf.cast(y_pred == i, tf.float32)

            self.true_positives.assign_add(tf.reduce_sum(y_true_i * y_pred_i))
            self.false_positives.assign_add(tf.reduce_sum((1 - y_true_i) * y_pred_i))
            self.false_negatives.assign_add(tf.reduce_sum(y_true_i * (1 - y_pred_i)))
            self.support.assign_add(tf.reduce_sum(y_true_i))
    
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        weighted_f1 = tf.reduce_sum((self.support / tf.reduce_sum(self.support)) * f1)
        return weighted_f1
    
    def reset_state(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))
        self.support.assign(tf.zeros_like(self.support))

# Upload of images for project
def load_npz_data(npz_path, resize):
    #Load train, val, and test arrays from a given .npz file.
    data = np.load(npz_path)
    
    # Extract images and labels
    train_images = data['train_images']
    train_labels = data['train_labels']
    val_images = data['val_images']
    val_labels = data['val_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

    # Ensure images are float32 and normalize to [0,1]
    train_images = train_images.astype("float32") / 255.0
    val_images = val_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    # Check if images are grayscale (1 channel) and convert to RGB (3 channels)
    if len(train_images.shape) == 3 or train_images.shape[-1] == 1:
        train_images = tf.image.grayscale_to_rgb(tf.expand_dims(train_images, axis=-1))
        val_images = tf.image.grayscale_to_rgb(tf.expand_dims(val_images, axis=-1))
        test_images = tf.image.grayscale_to_rgb(tf.expand_dims(test_images, axis=-1))

    # Resize if needed
    if resize:
        train_images = tf.image.resize(train_images, [32, 32], method='nearest')
        val_images = tf.image.resize(val_images, [32, 32], method='nearest')
        test_images = tf.image.resize(test_images, [32, 32], method='nearest')

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


# Data transformation
def create_tf_datasets(train_images, train_labels, val_images, val_labels, test_images, test_labels, num_classes, batch_size):
    """
    Creates TensorFlow datasets for training, validation, and testing.
    Also generates a stratified 30% subsample of the training set.
    """
    # Convert TensorFlow tensors to NumPy arrays for compatibility
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    val_images = np.array(val_images)
    val_labels = np.array(val_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    # Convert labels to categorical if num_classes > 2, else keep as binary
    if num_classes > 2:
        train_labels = to_categorical(train_labels, num_classes=num_classes)
        val_labels = to_categorical(val_labels, num_classes=num_classes)
        test_labels = to_categorical(test_labels, num_classes=num_classes)

    # Function to create TensorFlow datasets
    def create_dataset(images, labels, shuffle=False):
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(images))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    train_size = len(train_images)

    train_dataset = create_dataset(train_images, train_labels, shuffle=True)
    val_dataset = create_dataset(val_images, val_labels)
    test_dataset = create_dataset(test_images, test_labels)

    # Prepare labels for stratification (convert one-hot labels back to class indices)
    stratify_labels = np.argmax(train_labels, axis=1) if num_classes > 2 else train_labels

    # Create a stratified subsample

    if train_size > 80000:
        subsample_ratio = 0.3
    elif 10000 <= train_size <= 80000:
        subsample_ratio = 0.4
    else:
        subsample_ratio = 0.5

    train_images_sub, _, train_labels_sub, _ = train_test_split(
        train_images, train_labels, test_size=(1-subsample_ratio), stratify=stratify_labels, random_state=42
    )

    train_dataset_sub = create_dataset(train_images_sub, train_labels_sub, shuffle=True)

    return train_dataset, val_dataset, test_dataset, train_dataset_sub

def build_model(hp, num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    base_model.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(hp.Choice("units", [256, 512, 1024]), activation="relu")(x)
    x = Dropout(hp.Float("dropout", min_value=0.2, max_value=0.5, step=0.1))(x)
    
    activation = 'softmax' if num_classes > 2 else 'sigmoid'
    loss = 'categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    outputs = Dense(num_classes if num_classes > 2 else 1, activation=activation)(x)
    
    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    optimizer = {'adam': Adam, 'rmsprop': RMSprop, 'sgd': SGD}[optimizer_choice](learning_rate=learning_rate)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss, metrics=[WeightedF1Score(num_classes)])
    
    return model