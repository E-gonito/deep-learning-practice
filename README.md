# Deep Learning Practice

A repository for learning and experimenting with Deep Learning using Keras and TensorFlow, containerized with Docker for reproducibility and ease of use.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5d380431-accf-4fb4-83b5-cbb4ef399a58" alt="1ghkvvuoriz41">
</p>

Aigis - "Why don’t Dockerized models crash? Because they’ve got a *compose*d backup plan!"
---

## Overview

This project serves as a sandbox for building, testing, and exploring deep learning models. It leverages Keras and TensorFlow with GPU support, packaged in a Docker environment to simplify setup across systems. Key features include:

- A sample model implementation (`model.py`) with custom layers.
- GPU testing utilities (`test_gpu.py`).
- Docker-based workflow for consistent development and deployment.

---

## Requirements

To run this project, ensure you have the following installed:

- **Docker**: Container runtime for building and running the application. [Install Docker](https://docs.docker.com/get-docker/)
- **NVIDIA Drivers and Toolkit** (optional, for GPU support): Required if using the GPU-enabled TensorFlow image. [See TensorFlow GPU setup](https://www.tensorflow.org/install/gpu)
- **Python 3.11**: Included in the Docker image, no local installation needed.

Python dependencies are managed via `requirements.txt` and installed in the Docker container:

- TensorFlow (via base image)
- NumPy
- pydot (for model visualization)

For full details, see [TensorFlow Installation Guide](https://www.tensorflow.org/install/pip).

## Directory Structure
![Screenshot 2025-02-25 225408](https://github.com/user-attachments/assets/728a672e-465c-449e-bf10-3d9220a13df3)


## Setup and Usage

### Running the Project

1.  **Clone the Repository**:
    ```bash
    git clone
    cd learning_models
    ```
2.  **Set Environment Variables:**:
    ```bash
    export UID=$(id -u)
    export GID=$(id -g)
    ```
3.  **Start the Container:**:
    ```bash
    docker-compose up -d --build
    ```
4.  **Access the Container:**:
    ```bash
    docker-compose exec app bash
    ```
5.  **Running the script:**:
    ```bash
    python model.py
    ```

## Development Notes

- **GPU Support**: The `Dockerfile` uses `tensorflow/tensorflow:latest-gpu`. Ensure your host has NVIDIA drivers and `nvidia-container-toolkit` installed. [See TensorFlow GPU setup](https://www.tensorflow.org/install/gpu) for details.
- **Model Visualization**: Requires `pydot` and Graphviz. If `plot_model` fails, run the following inside the container:
  ```bash
  pip install pydot
  ```
