# Sentiment Analysis on Tweets

![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)
![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)
![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white) 
![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)
![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white) 

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Data Preprocessing](#data-preprocessing)
- [Training the Model](#training-the-model)
- [Results](#results)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)


## Introduction
This project is a sentiment analysis model applied to tweets. It uses machine learning and natural language processing (NLP) techniques to classify the sentiment of tweets into three categories: positive, negative, and neutral. The project demonstrates the use of various preprocessing steps, TF-IDF vectorization, and a deep learning model built with TensorFlow and Keras.

## Project Overview
Sentiment analysis is a key aspect of NLP that involves determining the sentiment expressed in a piece of text. This project leverages the power of deep learning and TF-IDF vectorization to perform sentiment analysis on a dataset of tweets.

## Features
- **Data Preprocessing**: Clean and preprocess raw tweet data, including removing stopwords and lemmatization.
- **TF-IDF Vectorization**: Convert the cleaned text data into numerical form using TF-IDF.
- **Deep Learning Model**: Train a neural network model to classify sentiments.
- **Visualization**: Plot accuracy and loss metrics to evaluate model performance.
  
## Model Architecture
The model used in this project is a sequential neural network with the following layers:

- **Dense Layer**: 100 units with ReLU activation
- **Dropout Layer**: 0.3 dropout rate
- **Dense Layer**: 25 units with ReLU activation
- **Dropout Layer**: 0.3 dropout rate
- **Dense Layer**: 10 units with ReLU activation
- **Dropout Layer**: 0.3 dropout rate
- **Output Layer**: 3 units with softmax activation for multiclass classification

## Data Preprocessing
The data preprocessing pipeline includes:

- **Lowercasing**: Convert all text to lowercase.
- **Contraction Expansion**: Expand contractions (e.g., "can't" to "cannot").
- **Special Character Removal**: Remove non-alphanumeric characters.
- **Tokenization**: Split text into words.
- **Stopword Removal**: Remove common stopwords.
- **Lemmatization**: Reduce words to their base or root form.

## Training the Model
The model is trained using the following configuration:

- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam with a learning rate of 0.01
- **Metrics**: Accuracy
- **Epochs**: 10

## Results
The performance of the model is evaluated using accuracy and loss plots. The final trained model is able to classify tweet sentiments with a reasonable accuracy.

![](https://github.com/Vidit-Kushwaha/Tweet-Sentimental-Analysis/blob/main/assets/result.png)

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

```bash
  git clone https://github.com/Vidit-Kushwaha/Tweet-Sentimental-Analysis.git
  cd Tweet-Sentimental-Analysis
```

2. Run the Jupyter Notebook:

```bash
  jupyter notebook Sentiment_Analysis_on_Tweets.ipynb
```

Train the model: Follow the steps in the notebook to preprocess the data, train the model, and visualize the results.

## Contributing

We welcome contributions! Whether you're a seasoned developer or a curious enthusiast, there are ways to get involved:

-   **Bug fixes and improvements:** Find any issues? Submit a pull request!
-   **New features:** Have an idea for a cool feature? Let's discuss it in an issue!
-   **Documentation:** Improve the project's documentation and website.
-   **Spread the word:** Share the project with your network and help it grow!

You can follow standard python contribution guidelines.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/Vidit-Kushwaha/Tweet-Sentimental-Analysis/blob/main/LICENSE) file for details.