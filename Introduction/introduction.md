# Introduction to ML
> “Machine Learning is the science (and art) of programming computers so they can learn from data”. [Aurélien Géron, 2019]


## How much data is there in the world?

Up until 2005: humans have created 130 Exabytes of data. How much is an Exabyte? (10^18)

- One letter = 1B

- One page of letters (2000 to 5000 letters), so half a page = 1KB

- One book of 500 pages = 1MB

- Human genome = 1GB

- I you take an HD camera and you follow a person for every single day of their life for every single hour minute and second and you film everything that they're doing for 70 or 80 years you can fit all of that material onto 1TB

- The Amazon rain forest takes up about 1.4 billion acres.That's 1.4 billion acres of trees. There's about 500 trees per acre making it about 700 billion trees. Now if you take all of these trees and you chop them down and you turn them into paper and you fill that paper with letters completely on both sides of the sheet then that will amount to somewhere between 0.5 and 1 petabyte (10^15)

Up until 2010: that number increased to 1200 Exabytes 

Up until 2015: 7900 Exabytes 

In 2020: probably something around **40900 Exabytes**.

# Types of ML Systems

## Trained with human supervision (or not)

### Supervised learning

Task: learning a function that maps an input to an output based on **example input-output pairs**.

#### Classification x Regression
Classification: Used to predict **discrete** values (class labels)

Regression: Used to predict **continuous** values

#### Important Supervised Learning Algorithms
* Linear Regression
* Logistic Regression
* k-Nearest Neighbors
* Support Vector Machines (SVMs)
* Neural Networks
* Decision Trees and Random Forests


### Unsupervised learning

Task: **inferring** the patterns within datasets without reference to known, or labeled, outcomes.

#### Clustering

Tries to detect similar groups.

#### Dimensionality reduction

Tries to simplify the data without losing too much information.

#### Important Unsupervised Learning Algorithms

* k-Means
* Hierarchical Cluster Analysis (HCA)
* Expectation Maximization
* Principal Component Analysis (PCA)
* Kernel PCA
* t-distributed Stochastic Neighbor Embedding (t-SNE)
* One-class SVM


### Reinforcement learning

Task: learning in an interactive environment by **trial and error**.

## Can learn incrementally on the fly (or not)

### Online Learning

Data becomes available in a sequential order, **predictors are updated at each step**.

### Batch Learning

Learns on the **entire training data set** at once.

## How they generalize

### Instance based learning

Learns the examples by heart and compares new problem instances with instances seen in training. EX: KNN

### Model based learning

All the assumptions about the problem domain are made explicit in the form of a model.


#  Chalenges in ML

## Bad data

* Insufficient quantity of training data
* Non representative training data
* Poor quality data (errors, outliers, noise)
* Irrelevant features (thus the importance of feature engineering)

## Bad algorithm

* Overfitting the training data
* Underfitting the training data

### Overfitting

The model performs well on the training data but it **does not generalize**. 
Overfitting happens when the model is too complex relative to the amount and noisiness of the training data.

### Underfitting

The model is **too simple** to learn the underlying structure of the data

# Validating and Testing

The only way to know how well a model will generalize to new cases is to actually try it out on new cases.

The data is usually split into a training set and a test set. It is common to use 80% of the data for training
and hold out 20% for testing.

## Cross-validation
Source: https://machinelearningmastery.com/k-fold-cross-validation/

Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample

The general procedure is as follows:

1. Shuffle the dataset randomly.
2. Split the dataset into k groups
3. For each unique group:
- Take the group as a hold out or test data set
- Take the remaining groups as a training data set
- Fit a model on the training set and evaluate it on the test set
- Retain the evaluation score and discard the model
4. Summarize the skill of the model using the sample of model evaluation scores