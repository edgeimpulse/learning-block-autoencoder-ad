import argparse, os, sys, random, logging, math
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras import layers


class AnomalyScoreLayer(layers.Layer):
    """Custom layer that computes anomaly score by comparing reconstruction error to threshold."""

    def __init__(self, threshold=0.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = tf.Variable(threshold, trainable=False, dtype=tf.float32, name='threshold')

    def call(self, inputs):
        # inputs is a tuple of (original_input, reconstruction)
        original, reconstruction = inputs
        # Compute MAE between original and reconstruction
        mae = tf.reduce_mean(tf.abs(original - reconstruction), axis=[1, 2])
        # Compute anomaly score as ratio of MAE to threshold
        # Score > 1.0 means anomaly, < 1.0 means normal
        anomaly_score = mae / self.threshold
        return anomaly_score

    def get_config(self):
        config = super().get_config()
        config.update({'threshold': float(self.threshold)})
        return config

# Lower TensorFlow log levels
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set random seeds for repeatable results
RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Load files
parser = argparse.ArgumentParser(description='Running custom Keras models in Edge Impulse')
parser.add_argument('--data-directory', type=str, required=False, default='data')
parser.add_argument('--epochs', type=int, required=False)
parser.add_argument('--learning-rate', type=float, required=False)
parser.add_argument('--out-directory', type=str, required=False, default='out')

args, unknown = parser.parse_known_args()

if not os.path.exists(args.out_directory):
    os.mkdir(args.out_directory)

# grab train/test set and convert into TF Dataset
x_np = np.load(os.path.join(args.data_directory, 'X_split_train.npy'), mmap_mode='r')
# X_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'), mmap_mode='r')


# Normalize in Studio instead
# # Normalize and save the mean and std we get,
# # for normalizing test data.
# training_mean = X_train.mean()
# training_std = X_train.std()
# X_train = (X_train - training_mean) / training_std
# print("Number of training samples:", len(X_train))

x_train = x_np

# print GPU/CPU info
print('Training on:', 'gpu' if len(tf.config.list_physical_devices('GPU')) > 0 else 'cpu')
print('')

# place to put callbacks (e.g. to MLFlow or Weights & Biases)
callbacks = []


print("Training input shape: ", x_np.shape)

# model architecture
# Calculate cropping needed: with two stride=2 layers, output may be larger than input
# due to ceiling division. Crop the excess from the end to match original input size.
INPUT_DIM = x_np.shape[1]
# Compute decoder output size: encoder does ceil(ceil(n/2)/2), decoder multiplies by 4
after_conv1 = math.ceil(INPUT_DIM / 2)
after_conv2 = math.ceil(after_conv1 / 2)
decoder_out = after_conv2 * 4
crop_amount = decoder_out - INPUT_DIM

model = keras.Sequential(
    [
        layers.Reshape((INPUT_DIM, 1), input_shape=(INPUT_DIM,)),
        layers.Conv1D(
            filters=32,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Conv1DTranspose(
            filters=16,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        # Crop output to match original input dimension (handles rounding from stride=2)
        layers.Cropping1D(cropping=(0, crop_amount)),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()

model.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ],
)

print('')
print('Training network OK')
print('')

# Get train MAE loss.
x_train_pred = model.predict(x_train)
train_mae_loss = np.mean(np.abs(x_train_pred.squeeze(-1) - x_train), axis=1)

# Get reconstruction loss threshold.
threshold = np.max(train_mae_loss)
print("Reconstruction error threshold: ", threshold)

# Create a new model with the anomaly score layer baked in
# Use batch_size=1 to ensure static tensor shapes for TFLite delegate compatibility
input_layer = layers.Input(shape=(x_np.shape[1],), batch_size=1, name='input')
reshaped_input = layers.Reshape((x_np.shape[1], 1))(input_layer)
reconstruction = model(reshaped_input)
anomaly_score = AnomalyScoreLayer(threshold=float(threshold), name='anomaly_score')([reshaped_input, reconstruction])
output = layers.Reshape((1,))(anomaly_score)

# Final model outputs a single anomaly score per sample
# Score > 1.0 indicates anomaly, < 1.0 indicates normal
final_model = keras.Model(inputs=input_layer, outputs=output)
final_model.summary()

# Save the model to disk
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
converter.experimental_new_converter = False  # Use legacy converter for older op versions
tflite_model = converter.convert()
with open(os.path.join(args.out_directory, 'model.tflite'), 'wb') as f:
    f.write(tflite_model)