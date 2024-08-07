# Amazon Review Sentiment Analyzer

# Overview
The Amazon Review Sentiment Analyzer is a Python-based application that analyzes the sentiment of Amazon product reviews. It uses machine learning techniques to predict and categorize the sentiment of reviews into Positive, Neutral, or Negative categories. The application connects to a MongoDB database to fetch and update review data.

# Features
- **Sentiment Analysis**: Predicts sentiment scores for product reviews.
- **Word Embeddings**: Creates word embeddings for reviews and uses them for sentiment prediction.
- **Data Processing**: Processes review data from MongoDB and updates the sentiment score and category.
- **Logging**: Provides detailed logging for debugging and monitoring the process.

# Prerequisites
- Python 3.6+
- MongoDB
- Git
- Python Packages: 'numpy', 'pymongo', 'logging'

# Installation

1. **Clone the Repository**
    
    git clone https://github.com/LFLGat/sentiment_analysis.git
    cd sentiment-analysis
    

2. **Create a Virtual Environment**
    
    python -m venv venv
    source venv/bin/activate  # On Windows use 'venv\Scripts\activate'
    

3. **Install Dependencies**
    
    pip install -r requirements.txt
    

4. **Set Up MongoDB**
    - Ensure you have MongoDB installed and running.
    - Update the 'MongoClient' connection string in the script with your MongoDB connection details.

# Usage

1. **Prepare Training Data**
    - Create a 'training.txt' file with review data and labels in the following format:
      
      review_text_1,label_1
      review_text_2,label_2
      

2. **Run the Application**
    
    python main.py
    

# Project Structure

amazon-review-sentiment-analyzer/
│
├── training.txt              # File containing training data
├── README.md                 # Project documentation
├── requirements.txt          # List of dependencies
├── main.py                   # Main application code
└── .gitignore                # Git ignore file


# Main Functions

# 'sigmoid(x)'
Calculates the sigmoid function of 'x'.

# 'predict(features, weights)'
Predicts the sentiment score using the given features and weights.

# 'cost_function(features, labels, weights)'
Calculates the cost function for logistic regression.

# 'update_weights(features, labels, weights, lr)'
Updates the weights using gradient descent.

# 'word_embedding(words, dimension)'
Creates word embeddings for the given words.

# 'vectorize_review(review, word_dict, dimension)'
Vectorizes a review using the given word dictionary.

# 'read_data(file_path)'
Reads review data from the specified file path.

# 'categorize_sentiment(score)'
Categorizes the sentiment based on the given score.

# 'train(features, labels, weights, lr, iters)'
Trains the model using the given features, labels, learning rate, and iterations.

# 'process_documents(word_dict, dimension, weights)'
Processes documents from the MongoDB collection and updates their sentiment scores and categories.

# Contributing

1. **Fork the Repository**
2. **Create a Feature Branch**
    sh
    git checkout -b feature-branch
    

3. **Commit Your Changes**
    sh
    git commit -m "Add new feature"
    

4. **Push to the Branch**
    sh
    git push origin feature-branch
    

5. **Open a Pull Request**

# Contact

Author: Joshua Gatmaitan

Email: joshgatmaitan286@gmail.com