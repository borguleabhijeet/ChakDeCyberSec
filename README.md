
# **Crime Report Classification**
This repository contains a project focused on classifying real-world crime reports into main and subcategories using machine learning and natural language processing (NLP) techniques. The dataset includes both English and Hinglish (Hindi written in English script) sentences, presenting unique challenges in text preprocessing and classification.



# **Project Description** 

## The project involves:
- Preprocessing: Cleaning the text data by removing special characters, handling Hinglish sentences, and preparing the dataset for classification.
- Text Vectorization: Using bigram vectorization to capture contextual information for classification tasks.
- Model Development: Implementing a neural network with a multi-task learning framework to classify main and subcategories.
- Exploratory Data Analysis (EDA): Understanding the dataset’s structure, patterns, and imbalances with visualizations and insights.

## Features



- Multi-task learning framework for simultaneous classification of main and subcategories.

- Extensive EDA including:

- Text length distribution.

- Category and subcategory distributions.

- N-gram analysis (bigram visualizations).

- Hinglish pattern detection.

- Model evaluation using metrics like accuracy, precision, recall, and F1-score.



## Installation



To run the project, ensure the following dependencies are installed in your Python environment:



## Requirements



- Python 3.x

- Pandas: For data manipulation.

- Scikit-learn: For feature extraction and preprocessing.

- Matplotlib: For visualizations.

- NLTK: For text preprocessing and stopword removal.

- TensorFlow: For building and training the neural network model.



## Install Dependencies



You can install the required libraries using pip:



```bash 
pip install pandas scikit-learn matplotlib nltk tensorflow 
```



How to Use



1. Clone the Repository:


```bash 

git clone https://github.com/your-username/crime-report-classification.git

cd crime-report-classification
```




2. Prepare Dataset:

- Place your dataset (CSV file) in the root directory.

- Ensure it contains the columns crimeaditionalinfo, category, and sub_category.

3. Run EDA:

Execute the provided EDA script to analyze the dataset and generate visualizations.


```bash 
python eda.py
python bi_gram.py
```




4. Train the Model:

Use the training script to preprocess data, vectorize text, and train the classification model.


```bash 
python train_model.py
```




5. Evaluate the Model:

Evaluate model performance on the test dataset, and analyze results for both main and subcategories.



## EDA Highlights



The EDA script provides the following insights:

- Category Distribution: Frequency of main and subcategories visualized using bar plots.

- Text Length Analysis: Distribution of word and character counts in the text.

- Bigram Analysis: Identifies common bigrams and visualizes them.

- Stopword Analysis: Ratio of stopwords to total words.

- Hinglish Patterns: Detects sentences containing Hinglish-specific words.



## Model Highlights



- Multi-task Learning: Separate classification heads for main and subcategories.

- Bigram Vectorization: Captures contextual features for better performance.

- Performance Metrics: Reports accuracy, precision, recall, and F1-score for both main and subcategories.



## Contributions



Feel free to contribute by:

- Improving preprocessing techniques for Hinglish sentences.

- Enhancing the multi-task learning model architecture.

- Providing better visualizations for EDA.



## License



This project is licensed under the MIT License. Feel free to use, modify, and distribute it.
