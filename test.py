import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the CSV data
csv_path = "alx/art_daily_jumpsup.csv"
df = pd.read_csv(csv_path)
data = df["value"].values.astype(np.float32)

print(f"Loaded {len(data)} samples from {csv_path}")

# Load TFLite model
model_path = "out/model.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get window size from model input shape
window_size = input_details[0]["shape"][1]
print(f"Model window size: {window_size}")
print(f"Input shape: {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")

# Normalize the data using training statistics
# We need to compute mean/std from a "normal" portion of data
# or use the first portion as reference
training_mean = data[:window_size].mean()
training_std = data[:window_size].std()
data_normalized = (data - training_mean) / training_std

# Slide window over the data and get anomaly scores
anomaly_scores = []
indices = []

for i in range(len(data) - window_size + 1):
    window = data_normalized[i : i + window_size]
    # Reshape to match model input: (1, window_size, 1)
    input_data = window.reshape(1, window_size).astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    anomaly_scores.append(output[0, 0])
    indices.append(i + window_size - 1)  # Index of the last sample in window

anomaly_scores = np.array(anomaly_scores)
indices = np.array(indices)

print(f"Computed {len(anomaly_scores)} anomaly scores")
print(f"Min score: {anomaly_scores.min():.4f}, Max score: {anomaly_scores.max():.4f}")
print(f"Scores > 1.0 (anomalies): {(anomaly_scores > 1.0).sum()}")

# Plot results
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Plot original data
axes[0].plot(data, color="#2E86AB", linewidth=0.8)
axes[0].set_ylabel("Value")
axes[0].set_title("Original Time Series Data")
axes[0].grid(True, alpha=0.3)

# Plot anomaly scores
axes[1].plot(indices, anomaly_scores, color="#A23B72", linewidth=0.8)
axes[1].axhline(y=1.0, color="#F18F01", linestyle="--", linewidth=1.5, label="Threshold (1.0)")
axes[1].set_ylabel("Anomaly Score")
axes[1].set_title("Anomaly Score (>1.0 = Anomaly)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot original with anomalies highlighted
axes[2].plot(data, color="#2E86AB", linewidth=0.8, label="Normal")
# Highlight anomalies
anomaly_mask = anomaly_scores > 1.0
if anomaly_mask.any():
    anomaly_indices = indices[anomaly_mask]
    axes[2].scatter(
        anomaly_indices,
        data[anomaly_indices],
        color="#C73E1D",
        s=10,
        label="Anomaly",
        zorder=5,
    )
axes[2].set_xlabel("Sample Index")
axes[2].set_ylabel("Value")
axes[2].set_title("Time Series with Anomalies Highlighted")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("anomaly_detection_output.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nPlot saved to anomaly_detection_output.png")

