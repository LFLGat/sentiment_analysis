import numpy as np
from pymongo import MongoClient
import logging
import random

# Connection to MongoDB
client = MongoClient('MONGO-CLIENT-KEY')
db = client.amazon
collection = db.amz

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Prediction function
def predict(features, weights):
    z = np.dot(features, weights)
    prediction = sigmoid(z)
    return np.atleast_1d(prediction)  # Ensure output is always an array


# Cost function
def cost_function(features, labels, weights):
    observations = len(labels)
    predictions = predict(features, weights)
    class1_cost = -labels*np.log(predictions)
    class2_cost = -(1-labels)*np.log(1-predictions)
    cost = class1_cost + class2_cost
    cost = cost.sum() / observations
    return cost

# Gradient descent function
def update_weights(features, labels, weights, lr):
    N = len(features)
    predictions = predict(features, weights)
    gradient = np.dot(features.T, predictions - labels) / N
    weights -= lr * gradient
    return weights


# Create a word embedding
def word_embedding(words, dimension=10):
    word_dict = {}
    for word in set(words):
        word_dict[word] = np.random.rand(dimension)
    return word_dict

# Average vectors for a review
def vectorize_review(review, word_dict, dimension=10):
    vectors = [word_dict[word] for word in review.split() if word in word_dict]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(dimension)

def read_data(file_path):
    reviews = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if ',' in line:
                review, label = line.rsplit(',', 1)
                reviews.append(review)
                labels.append(float(label))
            else:
                logging.warning(f"Skipping malformed line: {line}")
    return reviews, np.array(labels)


# Define a function to categorize sentiment based on the score
def categorize_sentiment(score):
    if score > .55:
        return 'Positive'
    elif score > .5:
        return 'Neutral'
    else:
        return 'Negative'

import logging

# Setup basic configuration for logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def train(features, labels, weights, lr, iters):
    for i in range(iters):
        logging.debug(f"Starting iteration {i}")
        weights = update_weights(features, labels, weights, lr)
        if i % 100 == 0:
            cost = cost_function(features, labels, weights)
            logging.info(f"Iteration: {i}, Cost: {cost}")
        logging.debug(f"Completed iteration {i}")
    return weights

def process_documents(word_dict, dimension, weights):
    processed_count = 0
    updated_count = 0
    logging.debug("Starting document processing")

    for document in collection.find():
        try:
            _id = document.get('_id', 'Unknown')
            review_text = document.get('reviews', {}).get('text', '')
            if not review_text:
                continue

            logging.debug(f"Processing document {_id}")
            features = vectorize_review(review_text, word_dict, dimension)
            sentiment_score = predict(features, weights)
            if isinstance(sentiment_score, np.ndarray) and sentiment_score.size > 0:
                sentiment_score = float(sentiment_score[0])  
            else:
                raise ValueError("Prediction did not return a valid array.")
            
            logging.debug(f"Predicted sentiment score: {sentiment_score}")
            sentiment_category = categorize_sentiment(sentiment_score)

            result = collection.update_one(
                {'_id': _id},
                {'$set': {
                    'reviews.sentimentScore': sentiment_score,
                    'reviews.sentimentCategory': sentiment_category
                }}
            )

            if result.modified_count > 0:
                updated_count += 1
            processed_count += 1
            logging.info(f"Processed document {_id}: Sentiment score {sentiment_score}, Sentiment category {sentiment_category}")

        except Exception as e:
            logging.error(f"Error processing document {_id}: {e}")

    logging.info(f"Processing completed. Total documents processed: {processed_count}. Documents updated: {updated_count}.")
    logging.debug("Document processing completed")


def main():
    reviews, labels = read_data('training.txt')
    dimension = 10
    word_dict = word_embedding(" ".join(reviews).split(), dimension)
    features = np.array([vectorize_review(review, word_dict, dimension) for review in reviews])
    weights = np.random.rand(dimension)
    weights = train(features, labels, weights, lr=0.05, iters=5000)
    process_documents(word_dict, dimension, weights)

if __name__ == "__main__":
    main()
