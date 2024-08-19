# Bert_Fine_Tuning

# IMDB Sentiment Analysis with BERT

This basic project demonstrates how to use the BERT model for sentiment analysis on the IMDB dataset. The project uses the Hugging Face `transformers` library to tokenize the dataset, train the model, and evaluate its performance.

## Project Structure

- `train_model.py`: Python script to train the BERT model on the IMDB dataset.
- `requirements.txt`: List of dependencies required to run the project.
- `README.md`: Project overview and setup instructions.

## Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/imdb-sentiment-analysis.git
    cd imdb-sentiment-analysis
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the training script**:
    ```bash
    python train_model.py
    ```

This script will download the IMDB dataset, tokenize the text using BERT tokenizer, and train a sequence classification model using BERT. The training results will be saved in the `results/` directory, and logs will be stored in the `logs/` directory.


This project demonstrates how to use the BERT model for sentiment analysis on the IMDB dataset. The project uses the Hugging Face `transformers` library to tokenize the dataset, train the model, and evaluate its performance.
# IMDB Sentiment Analysis with BERT

This project demonstrates how to use the BERT model for sentiment analysis on the IMDB dataset. The project uses the Hugging Face `transformers` library to tokenize the dataset, train the model, and evaluate its performance.

## Project Structure
Bert_Fine_Tuning/
│
├── train_model.py  Script for training the BERT model
├── requirements.txt  List of dependencies
└── README.md  Project overview and setup instructions

## Project Explanation

The project is a basic demonstration of fine-tuning BERT for sentiment analysis. The IMDB dataset is used, which contains 50,000 movie reviews labeled as positive or negative. The model is trained for two epochs using the Hugging Face Trainer API, which simplifies the training process.
