# 1D Convolutional Autoencoder for Time-Series Anomaly Detection

This repository contains a 1D convolutional autoencoder for anomaly detection on unprocessed time-series data. The model is trained on nominal data only and uses reconstruction error to identify anomalies, avoiding the need for manual feature engineering.

It can be integrated into an Edge Impulse pipeline to:

- Replace traditional methods like k-means or GMM
- Train directly on raw time-series data
- Deploy to edge devices using the standard Edge Impulse workflow

## Edge Impulse integration

This model is designed to run as a custom learning block in Edge Impulse, paired with the raw data processing block. See the [Custom blocks](https://docs.edgeimpulse.com/studio/organizations/custom-blocks) and [Custom learning blocks](https://docs.edgeimpulse.com/studio/organizations/custom-blocks/custom-learning-blocks) documentation for additional details.

Following the steps below makes the block available under any of your projects, just like any other learning block, so you can add it to your impulse and deploy the model to an edge device.

1. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/tools/clis/edge-impulse-cli/installation)
2. Clone this repo
3. Change into the block directory
4. Initialize the block:

```
$ edge-impulse-blocks init --clean
```

5. Push the block to Edge Impulse:

```
$ edge-impulse-blocks push
```

## Local pipeline integration

You can run this pipeline locally with Docker. This encapsulates all dependencies and packages for you. 

Following the steps below will train the model and create a saved model ZIP file in the `out` directory.

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/tools/clis/edge-impulse-cli/installation)
3. Create a new Edge Impulse project and add data
4. Under Create impulse add a Raw data processing block and any anomaly detection learning block
5. Clone this repo
6. Change into the block directory
7. Initialize the block:

```
$ edge-impulse-blocks init --clean
```

6. Fetch new data:

```
$ edge-impulse-blocks runner --download-data data/
```

7. Build the container:

```
$ docker build -t custom-ml-autoencoder-ad .
```

8. Run the container:

_macOS / Linux_

```
$ docker run --rm -v $PWD:/app custom-ml-autoencoder-ad --data-directory /app/data --epochs 50 --learning-rate 0.001 --out-directory out/
```

_Windows (Command prompt)_

```
$ docker run --rm -v "%cd%":/app custom-ml-autoencoder-ad --data-directory /app/data --epochs 50 --learning-rate 0.001 --out-directory out/
```

_Windows (Powershell)_

```
$ docker run --rm -v ${PWD}$:/app custom-ml-autoencoder-ad --data-directory /app/data --epochs 50 --learning-rate 0.001 --out-directory out/
```
