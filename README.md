# TFX ML Production Pipeline for Sentiment Analysis

## Overview
This project demonstrates how to build and deploy a production machine learning pipeline using TensorFlow Extended (TFX) by using BERT model. The pipeline is designed for sentiment analysis of movie reviews, transforming raw text data into a trained model that can predict sentiment. 🎥📊

## Prerequisites
- Python 3.x
- Google Colab or a local Jupyter Notebook setup
- Internet access for downloading datasets and dependencies
- Docker 🐳

## Installation
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/HarshaRockzz/TFX-ML-Production-Pipeline-for-Sentiment-Analysis.git
    cd TFX-ML-Production-Pipeline-for-Sentiment-Analysis
    ```

2. **Install Required Packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Setup
### Google Colab
If you are using Google Colab, ensure to install the required packages each time you start the notebook:

```python
!pip install tfx
!pip install apache-beam[gcp]
!pip install tensorflow
!pip install tensorflow-data-validation
!pip install tensorflow-transform
!pip install tensorflow-model-analysis
!pip install tensorflow-serving-api
!pip install tensorboard
!pip install ngrok
```

### Local Setup
For a local setup, ensure all dependencies listed in requirements.txt are installed.

## Usage
Run the Pipeline:

1. Open tfx.ipynb in Google Colab or Jupyter Notebook.
2. Follow the steps outlined in the notebook to run the TFX pipeline components.

### Data Preparation:
- Ensure your dataset is in the correct format.
- Modify the data loading section if you are using a different dataset.

### Pipeline Components:
The pipeline consists of the following components:
- ExampleGen: Ingests data into the pipeline.
- StatisticsGen: Generates statistics for data visualization.
- SchemaGen: Generates a schema based on the data statistics.
- ExampleValidator: Detects anomalies in the data.
- Transform: Preprocesses and transforms the data.
- Trainer: Trains the machine learning model.
- Tuner: Performs hyperparameter tuning to optimize model performance.
- Evaluator: Evaluates the model performance.
- Pusher: Pushes the model to a serving infrastructure.

## Hyperparameter Tuning
The Tuner component uses Keras Tuner to find the best hyperparameters for the model.
You can configure the hyperparameters search space in the tuner_fn function.

## Drift Detection and Model Retraining
The pipeline includes drift detection to monitor data drift.
If drift is detected, the pipeline automatically retrains the model on a new dataset.
The ExampleValidator component helps in detecting data drift by comparing the current data with a reference dataset.

## Visualization with TensorBoard
You can visualize the training process and evaluation metrics using TensorBoard.
Use ngrok to create a secure tunnel to access TensorBoard on Google Colab:

```python
from tensorboard import notebook
import ngrok

# Start TensorBoard
%load_ext tensorboard
%tensorboard --logdir=logs/

# Start ngrok
!ngrok http 6006
```

## Metadata Containerization with Docker
The metadata generated by the TFX pipeline can be containerized using Docker.
- Build a Docker image:
```bash
docker build -t tfx_pipeline_metadata .
```
- Run the Docker container:
```bash
docker run -p 8080:8080 tfx_pipeline_metadata
```

## Model Training
The Trainer component is responsible for training the sentiment analysis model. Ensure the training data is properly preprocessed before initiating the training process.

## Evaluation and Deployment
Use the Evaluator component to assess the model's performance.
The Pusher component can deploy the model to TensorFlow Serving for production use.

## Acknowledgements
- TensorFlow and TFX documentation for the detailed instructions and examples.
