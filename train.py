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
parser.add_argument('--epochs', type=int, required=False, default=50)
parser.add_argument('--learning-rate', type=float, required=False, default=0.001)
parser.add_argument('--kernel-size', type=int, required=False, default=7, choices=[3, 5, 7])
parser.add_argument('--conv-layers', type=int, required=False, default=2, choices=range(1, 7))
parser.add_argument('--out-directory', type=str, required=False, default='out')
parser.add_argument('--early-stopping-patience', type=int, required=False, default=5)

args, unknown = parser.parse_known_args()

if not os.path.exists(args.out_directory):
    os.mkdir(args.out_directory)

# grab train/test set and convert into TF Dataset
x_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'), mmap_mode='r')

# At time of publish, X_split_test is written, but with 0 samples
# x_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'), mmap_mode='r')

# print GPU/CPU info
print('Training on:', 'gpu' if len(tf.config.list_physical_devices('GPU')) > 0 else 'cpu')
print('')


print("Training input shape: ", x_train.shape)
# print("Testing input shape", x_test.shape) # Would be (0,x)

KERNEL_SIZE = args.kernel_size
CONV_LAYERS = args.conv_layers

# model architecture
# Calculate cropping needed: with two stride=2 layers, output may be larger than input
# due to ceiling division. Crop the excess from the end to match original input size.
INPUT_DIM = x_train.shape[1]
# Compute decoder output size: encoder does repeated ceil(n/2), decoder multiplies by 2^CONV_LAYERS
after_conv = INPUT_DIM
for _ in range(CONV_LAYERS):
    after_conv = math.ceil(after_conv / 2)
decoder_out = after_conv * (2 ** CONV_LAYERS)
crop_amount = decoder_out - INPUT_DIM

# Encoder/decoder channel plan.
# For CONV_LAYERS=2, this yields [32, 16] (matching the previous architecture).
# For CONV_LAYERS>=3, this starts at 64 (e.g., [64, 32, 16, ...]).
base_filters = 64 if CONV_LAYERS >= 3 else 32
encoder_filters = [max(4, base_filters // (2 ** i)) for i in range(CONV_LAYERS)]

model_layers = [layers.Reshape((INPUT_DIM, 1), input_shape=(INPUT_DIM,))]

for i, filters in enumerate(encoder_filters):
    model_layers.append(
        layers.Conv1D(
            filters=filters,
            kernel_size=KERNEL_SIZE,
            padding="same",
            strides=2,
            activation="relu",
        )
    )
    if i < CONV_LAYERS - 1:  # No dropout after last encoder layer
        model_layers.append(layers.Dropout(0.2))

# Mirror decoder: same number of upsampling transposed convs as encoder convs
for i, filters in enumerate(reversed(encoder_filters)):
    model_layers.append(
        layers.Conv1DTranspose(
            filters=filters,
            kernel_size=KERNEL_SIZE,
            padding="same",
            strides=2,
            activation="relu",
        )
    )
    if i != 0:
        model_layers.append(layers.Dropout(0.2))

# Final reconstruction layer (no upsampling): total Conv1DTranspose layers = CONV_LAYERS + 1
model_layers.append(layers.Conv1DTranspose(filters=1, kernel_size=KERNEL_SIZE, padding="same"))

# Crop output to match original input dimension (handles rounding from stride=2)
model_layers.append(layers.Cropping1D(cropping=(0, crop_amount)))

model = keras.Sequential(model_layers)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate), loss="mse")
model.summary()

dataset_size = x_train.shape[0]
if dataset_size < 10000:
    validation_split = 0.2
elif dataset_size < 1000000:
    validation_split = 0.1
else:
    validation_split = 0.01

print(f'Size is {dataset_size}, setting validation split to {validation_split}')

model.fit(
    x_train,
    x_train, # Target matches for autoencoders
    epochs=args.epochs,
    batch_size=128,
    validation_split=validation_split,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience, mode="min"),
        tf.keras.callbacks.TensorBoard(log_dir="/home/tensorboard_logs")
    ],
)

print('')
print('Training network OK')
print('')

# Get train MAE loss.
x_train_pred = model.predict(x_train)
train_mae_loss = np.mean(np.abs(x_train_pred.squeeze(-1) - x_train), axis=1)

# Get reconstruction loss threshold.
# I'm dividing the reconstruction MAE by this mean from training to sort of normalize the anomaly score.
threshold = np.max(train_mae_loss)
print("Reconstruction error threshold: ", threshold)

# Create a new model with the anomaly score layer baked in
# Use batch_size=1 to ensure static tensor shapes for TFLite delegate compatibility
input_layer = layers.Input(shape=(x_train.shape[1],), batch_size=1, name='input')
reshaped_input = layers.Reshape((x_train.shape[1], 1))(input_layer)
reconstruction = model(reshaped_input)
anomaly_score = AnomalyScoreLayer(threshold=float(threshold), name='anomaly_score')([reshaped_input, reconstruction])
output = layers.Reshape((1,))(anomaly_score)

# Final model outputs a single anomaly score per sample
final_model = keras.Model(inputs=input_layer, outputs=output)
final_model.summary()

# Save the model to disk
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
converter.experimental_new_converter = False  # Use legacy converter for older op versions
tflite_model = converter.convert()
with open(os.path.join(args.out_directory, 'model.tflite'), 'wb') as f:
    f.write(tflite_model)