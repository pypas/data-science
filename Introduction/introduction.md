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


*OBS: an epoch is one **stream** of our entire dataset. The number of epochs we define is the amount of times our model will see the entire dataset* 

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

Corresponds to a situation of **high variance**. Variance is how much the prediction function would change if we estimated it using a different training set. If the variance is high, it means that small changes in the training set lead to large changes in our prediction function. Usually, more flexible methods have higher variance.

### Underfitting

The model is **too simple** to learn the underlying structure of the data.

Corresponds to a situation of **high bias**. Bias is the error introduced by approximating a real life problem by a simple model. Ex: linear regression assumes a linear relationship.

### The bias/variance tradeoff
A model’s generalization error can be expressed as the sum of three errors: bias, variance and irreductible error.

- Increasing a model’s complexity will typically increase its variance and reduce its bias.

- Reducing a model’s complexity increases its bias and reduces its variance. 

#### Bias
- Due to wrong assumptions, such as assuming that the data is linear when it is actually quadratic.

- A high-bias model is most likely to **underfit** the training data.

- $J_{train}(\Theta)$ will be high (since the fit is not good) and $J_{cv}(\Theta) \approx $J_{train}(\Theta)$

#### Variance
- Due to the model’s excessive sensitivity to small variations in the training data.

- A model with many degrees of freedom is likely to have high variance, and thus to **overfit** the training data

- $J_{train}(\Theta)$ will be low (since the fit is good on the training data) and $J_{cv}(\Theta) >> $J_{train}(\Theta)$ (due to the overfit)

#### Irreductible error
- Due to the noisiness of the data itself.

- The only way to reduce this part of the error is to **clean up the data**.

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

## Testing and Error Metrics

### Accuracy
Out of all the data, how many points did we classify correctly?

### Precision
Out of all the points we’ve **predicted to be positive**, how many are **correct**?

Ex: for a Spam Detector, we need a high precision
- False positives NOT ok (ex: sending an email that was not spam to the spam folder)

- False negatives ok (ex: sending a spam to the inbox folder)

### Recall
Out of all the points **labelled positive**, how many did we **correctly predict**?

Ex: for a Medical Model (predicting if a patient is sick or not), we need a high recall
- False positives ok (ex: patient diagnosed sick, but is healthy)

- False negatives NOT ok (ex: patient diagnosed healthy, but is sick)

### F1 Score
**Harmonic mean** of precision and recall.

OBS: $h_mean(x,y) = \frac{2xy}{x+y}$

### F$\Beta$ Score
We choose beta according to what we want to prioritize (precision or recall). 

$F\Beta = (1+\Beta)^2\frac{precision\*recall}{\Beta^2\*precision+recall}$

## AI x Machine Learning 

Artificial Intelligence: The effort to automate intellectual tasks normally performed by humans. Machine Learning is a part of AI.

Machine Learning: figure out the rules for AI. 

Neural Networks: form of Machine Learning that uses a layered representation of data. Neural networks are not modeled after the way the human brain works.