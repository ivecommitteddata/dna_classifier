
# 🧬 DNA Profile Classification with Python 🐍

## Introduction

In this tutorial, we'll delve into using Python and its robust libraries to classify DNA profiles. DNA profiles, which are unique sets of genetic markers tied to an individual, are instrumental in various domains, including forensic science, to pinpoint suspects.

We'll employ the K-Nearest Neighbors algorithm, a renowned machine learning methodology, to categorize the DNA profiles.

## Prerequisites

- Python
- pandas: A data analysis library
- scikit-learn: A machine learning library
- numpy: A library for numerical computations in Python

To install the essential libraries, use pip:

```bash
pip install pandas scikit-learn numpy
```

## Data Preparation

Our dataset, `dna_profiles.csv`, is replete with DNA profiles of distinct individuals. Each row delineates an individual's DNA profile, characterized by genetic markers labeled as A, C, G, and T.

```python
import pandas as pd

# Load the DNA profiles data into a Pandas dataframe
dna_profiles = pd.read_csv("dna_profiles.csv")
```

## Data Splitting

The dataset will be partitioned into two segments:
1. Training data: This is used to educate our classifier.
2. Testing data: This segment is leveraged to appraise the classifier's efficacy.

```python
# Split the data into training and testing sets
train_size = int(0.8 * len(dna_profiles))
training_data = dna_profiles[:train_size]
testing_data = dna_profiles[train_size:]
```

## Constructing the Classifier

For our endeavor, the K-Nearest Neighbors classifier from scikit-learn is the chosen tool. This classifier will be nurtured using the training data.

```python
from sklearn.neighbors import KNeighborsClassifier

# Train the K-Nearest Neighbors classifier
classifier = KNeighborsClassifier()
classifier.fit(training_data[["A", "C", "G", "T"]], training_data["Label"])
```

## Classifier Evaluation

Post-training, the classifier's prowess is gauged using the testing data.

```python
# Test the classifier on the testing data
predictions = classifier.predict(testing_data[["A", "C", "G", "T"]])

# Collate the predictions with the actual labels
correct_predictions = sum(predictions == testing_data["Label"])
accuracy = correct_predictions / len(testing_data)
print("Accuracy:", accuracy)
```

## Conclusion

By harnessing Python, pandas, scikit-learn, and numpy, we've adeptly classified DNA profiles. This methodology can be amplified and refined to synergize with expansive datasets and more intricate algorithms.
