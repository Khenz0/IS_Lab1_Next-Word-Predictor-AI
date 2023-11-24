# IS_Lab1_Next-Word-Predictor-AI
## Text Generation using LSTM

## Overview
This repository contains a Python Jupyter notebook demonstrating the implementation of a Long Short-Term Memory (LSTM) neural network for text generation. The model is trained on a dataset of news articles to predict the next word in a sequence of words. Additionally, a text generation function is provided to generate new text based on an initial input.

## Prerequisites
- Python 3.x
- Required Python libraries: `numpy`, `pandas`, `matplotlib`, `nltk`, and `tensorflow`

## Setup
1. Ensure you have the required Python libraries installed. You can install them using the following command:
```python
pip install numpy pandas matplotlib nltk tensorflow
```

2. Download the news dataset (`news.csv`) and place it in the same directory as this notebook.

## Usage
1. Open the Jupyter notebook (`text_generation.ipynb`) in your Jupyter environment.

2. Run the notebook cell by cell.

3. If you want to start training the model from scratch, execute all cells. If you want to load a pre-trained model and generate text, skip to the "Loading the Model" section.

4. Once the model is trained or loaded, you can use the provided functions for text prediction and generation. See the "Example Usage" section for demonstrations.

## Files
- `text_generation.ipynb`: The Jupyter notebook containing the code.
- `news.csv`: The dataset containing news articles (not provided).

## Example Usage
1. **Training the Model:** Execute cells under the "Training the Model" section to train the LSTM model.

2. **Text Prediction:** Utilize the `predict_next_word` function to predict the next word(s) given an input text.

 ```python
 possible = predict_next_word("She will have to look into this thing and she", 5)
 for idx in possible:
     print(unique_tokens[idx])
 ```

3. **Text Generation:** Use the `generate_text` function to generate text based on an initial input.

 ```python
 generated_text = generate_text("He must have one thing that I am into the", 100, 10)
 print(generated_text)
 ```

## Model and History Files
- `text_gen_model1.h5`: The trained LSTM model.
- `history1.p`: Pickled file containing the training history.

## Notes
- Make sure to adjust the file paths if your dataset is stored in a different location.

Feel free to experiment with different parameters and dataset sizes for training the model.
