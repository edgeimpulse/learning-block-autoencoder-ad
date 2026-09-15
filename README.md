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
4. Under **Create impulse** add a 'Raw data' processing block, and the "Anomaly detection" (without K-means / GMM).
5. Open a command prompt or terminal window.
6. Initialize the block:

    ```bash
    $ edge-impulse-blocks init --clean
    ```

7. Fetch new data via:

    ```bash
    $ edge-impulse-blocks runner --download-data data/
    ```

8. Build the container:

    ```bash
    $ docker build -t custom-ml-autoencoder-ad .
    ```

9. Run the container to test the script (you don't need to rebuild the container if you make changes):

    **macOS, Linux**

    ```bash
    $ docker run --rm -v $PWD:/app custom-ml-autoencoder-ad --data-directory /app/data --epochs 50 --learning-rate 0.001 --out-directory out/
    ```

    **Windows (Command prompt)**

    ```bash
    $ docker run --rm -v "%cd%":/app custom-ml-autoencoder-ad --data-directory /app/data --epochs 50 --learning-rate 0.001 --out-directory out/
    ```

    **Windows (Powershell)**

    ```bash
    $ docker run --rm -v ${PWD}$:/app custom-ml-autoencoder-ad --data-directory /app/data --epochs 50 --learning-rate 0.001 --out-directory out/
    ```

    > **Training on GPU:** If you're on an x86 machine with an Nvidia GPU, you can pass `--gpus=all` (right after `docker run`) to train on GPU.

10. This creates a TFLite file in the 'out' directory.

#### Adding extra dependencies

If you have extra packages that you want to install within the container, add them to `requirements.txt` and rebuild the container.

#### Adding new arguments

To add new arguments, see [Custom learning blocks > Arguments to your script](https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/adding-custom-learning-blocks#arguments-to-your-script).

## Fetching new data

To get up-to-date data from your project:

1. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) v1.16 or higher.
2. Open a command prompt or terminal window.
3. Fetch new data via:

    ```
    $ edge-impulse-blocks runner --download-data data/
    ```

## Pushing the block back to Edge Impulse

You can also push this block back to Edge Impulse, that makes it available like any other ML block so you can retrain your model when new data comes in, or deploy the model to device. See [Docs > Adding custom learning blocks](https://docs.edgeimpulse.com/docs/edge-impulse-studio/organizations/adding-custom-transfer-learning-models) for more information.

1. Push the block:

    ```
    $ edge-impulse-blocks push
    ```

2. The block is now available under any of your projects via **Create impulse > Add new learning block**.
