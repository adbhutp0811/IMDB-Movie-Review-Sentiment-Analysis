# IMDB Movie Review Sentiment Analysis

A deep learning project that classifies movie reviews as **positive** or **negative** using a Recurrent Neural Network (RNN) trained on the IMDB dataset, served through a Streamlit web application.

## Overview

The project uses a Simple RNN model built with TensorFlow/Keras to perform binary sentiment classification on text reviews. The workflow is split into training, experimentation, and deployment:

- `RNN.ipynb` — loads and preprocesses the IMDB dataset, builds and trains the model, then saves it.
- `prediction.ipynb` — explores loading the saved model and making predictions on sample reviews.
- `main.py` — a Streamlit app that provides an interactive UI for classifying user-written reviews.
- `simple_rnn_imdb.h5` — the pre-trained model file.

## Model Architecture

The model is a `Sequential` network with the following layers:

| Layer | Configuration |
| --- | --- |
| Input | shape `(500,)` — padded review sequences |
| Embedding | `10,000` words, `128`-dimensional vectors |
| SimpleRNN | `128` units, `tanh` activation |
| Dense | `1` unit, `sigmoid` activation (binary output) |

**Training configuration:**

- Dataset: Keras IMDB dataset, limited to the top `10,000` words
- Sequences padded/truncated to a maximum length of `500`
- Optimizer: `adam` · Loss: `binary_crossentropy` · Metric: `accuracy`
- `10` epochs, `batch_size=32`, `validation_split=0.2`
- `EarlyStopping` on `val_loss` (patience `5`, restore best weights)

## Project Structure

```
.
├── main.py                  # Streamlit web application
├── prediction.ipynb         # Prediction experiments / demo
├── RNN.ipynb                # Model training notebook
├── simple_rnn_imdb.h5       # Trained model weights
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Installation

1. Clone the repository and navigate into the project directory.

2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the web app

```bash
streamlit run main.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`), enter a movie review in the text area, and click **Classify Sentiment**. The app displays the predicted sentiment, a prediction score, and a progress bar.

### Retrain the model

Open and run `RNN.ipynb` in Jupyter Notebook/JupyterLab. It downloads the IMDB dataset, trains the model with early stopping, and saves the result to `simple_rnn_imdb.h5`.

### Try predictions in a notebook

Open `prediction.ipynb` and run the cells to decode reviews, preprocess text, and classify a sample review.

## How It Works

1. Input text is tokenized into words using `text_to_word_sequence`.
2. Each word is mapped to its IMDB word index, offset by `+3` (indices `0`, `1`, and `2` are reserved for padding, start, and unknown tokens). Unknown or out-of-vocabulary words are encoded as `2`.
3. The encoded sequence is padded/truncated to length `500`.
4. The model outputs a value between `0` and `1`; scores `>= 0.5` are classified as **Positive**, otherwise **Negative**.

## Dependencies

- streamlit
- tensorflow
- numpy
- pandas
- scikit-learn
- nltk
