import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Load the DNA profiles data into a Pandas dataframe
dna_profiles = pd.read_csv("dna_profiles.csv")

# Split the data into training and testing sets
train_size = int(0.8 * len(dna_profiles))
training_data = dna_profiles[:train_size]
testing_data = dna_profiles[train_size:]

# Ensures there's data in both training and testing datasets
if len(training_data) == 0 or len(testing_data) == 0:
    raise ValueError("Insufficient data for training or testing. Ensure your CSV has enough records.")

# Ensure n_neighbors is less than or equal to the number of training samples
num_neighbors = min(5, len(training_data))

# Create the classifier with the specified number of neighbors
classifier = KNeighborsClassifier(n_neighbors=num_neighbors)

# Train the classifier using the DNA profile markers as features and "Label" as the target
classifier.fit(training_data[["A", "C", "G", "T"]], training_data["Label"])

# Test the classifier on the testing data
predictions = classifier.predict(testing_data[["A", "C", "G", "T"]])

# Compare the predictions to the actual labels
correct_predictions = sum(predictions == testing_data["Label"])
accuracy = correct_predictions / len(testing_data)
print("Accuracy:", accuracy)
