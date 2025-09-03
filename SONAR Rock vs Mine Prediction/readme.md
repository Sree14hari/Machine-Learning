# Sonar Rock vs. Mine Prediction using Logistic Regression

## Project Overview

This project implements a binary classification model using Logistic Regression to distinguish between sonar signals bounced off a metal cylinder (representing a "mine") and those bounced off a rock. The model is trained on the "Sonar, Mines vs. Rocks" dataset, which contains sonar returns from various angles. The goal is to build a predictive system that can accurately classify an object as either a rock or a mine based on its sonar signature.

## Dataset

The dataset used is the **Sonar, Mines vs. Rocks** dataset, loaded from a `sonar.csv` file.

  * **Instances:** The dataset contains 208 instances (rows).
  * **Features:** There are 60 feature columns. Each column represents the energy in a particular frequency band, integrated over a certain period of time. These values range from 0.0 to 1.0.
  * **Target Variable:** The 61st column is the target label.
      * '**R**' indicates the object is a **Rock**.
      * '**M**' indicates the object is a **Mine**.

The dataset is fairly balanced, with 111 instances labeled as 'M' (Mine) and 97 as 'R' (Rock).

## Dependencies

This project uses the following Python libraries. You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

  -**Pandas:** For data manipulation and loading the CSV file.
  -**NumPy:** For numerical operations, especially for handling the input data for prediction.
  -**Matplotlib & Seaborn:** For data visualization (though not used for plotting in the final version of the notebook).
  -**Scikit-learn:** For building and evaluating the machine learning model.

## Methodology

The project follows a standard machine learning workflow, from data exploration to building a predictive system.

### 1\. Data Loading and Initial Inspection

First, the necessary libraries are imported. The dataset is loaded from `Dataset/sonar.csv` into a Pandas DataFrame. Since the CSV file does not have a header, `header=None` is specified.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Loading the dataset
data = pd.read_csv("Dataset/sonar.csv", header=None)
```

  -`data.head()` is used to view the first 5 rows and get a feel for the data.
  -`data.shape` is used to confirm the dimensions of the dataset (208 rows, 61 columns).

### 2\. Data Preprocessing and Exploratory Data Analysis (EDA)

This step involves understanding the data's characteristics and preparing it for the model.

  -**Checking for Missing Values:**
    ```python
    data.isnull().sum()
    ```
    The output confirms that there are no null or missing values in the dataset, so no imputation is needed.

  -**Statistical Summary:**
    ```python
    data.describe()
    ```
    This command generates descriptive statistics for each feature column, such as mean, standard deviation, min, max, and quartile values. This provides insight into the distribution of the sonar signal data.

  -**Class Distribution Analysis:**
    ```python
    data[60].value_counts()
    ```
    This shows the number of samples for each class ('M' and 'R'), confirming the dataset is reasonably balanced.

### 3\. Feature and Target Separation

The dataset is split into features (`x`) and the target label (`y`). Column 60 contains the labels, while columns 0-59 are the features.

```python
x = data.drop(60, axis=1) # Features
y = data[60]              # Target
```

### 4\. Splitting Data into Training and Testing Sets

The data is divided into a training set and a testing set. The model will learn from the training set and be evaluated on the unseen testing set.

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=1)
```

  -`test_size=0.1`: 10% of the data is reserved for testing, and 90% for training.
  -`stratify=y`: This is a crucial parameter. It ensures that the proportion of 'Rock' and 'Mine' samples is the same in both the training and testing sets as it is in the original dataset. This prevents class imbalance issues during evaluation.
  -`random_state=1`: This ensures that the data is split in the same way every time the code is run, making the results reproducible.

### 5\. Model Training

A Logistic Regression model is chosen for this classification task.

```python
# Initialize the model
lr = LogisticRegression()

# Train the model with the training data
lr.fit(x_train, y_train)
```

The `fit` method trains the Logistic Regression algorithm on the training features (`x_train`) and their corresponding labels (`y_train`).

### 6\. Model Evaluation

The model's performance is evaluated using the accuracy score, which measures the proportion of correctly classified instances.

  -**Accuracy on Training Data:**
    ```python
    x_pred = lr.predict(x_train)
    train_accu = accuracy_score(x_pred, y_train)
    # Result: ~83.4%
    ```
    This shows how well the model fits the data it was trained on.

  -**Accuracy on Test Data:**
    ```python
    x_pred_test = lr.predict(x_test)
    test_accu = accuracy_score(x_pred_test, y_test)
    # Result: ~76.2%
    ```
    This is the more important metric as it reflects the model's ability to generalize to new, unseen data. An accuracy of 76.2% indicates a reasonably effective model.

### 7\. Building a Predictive System

Finally, a predictive system is created to demonstrate how the trained model can be used on a single new data point.

```python
# Example input data for a "Mine"
input_data = (0.0530, ..., 0.0056)

# Convert to a NumPy array
input_numpy = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
reshaped_data = input_numpy.reshape(1, -1)

# Make a prediction
prediction = lr.predict(reshaped_data)

if prediction[0] == "R":
    print("The object is a Rock")
else:
    print("The object is a Mine")
```

This script takes a tuple of 60 feature values, converts it into the format expected by the model, and outputs a human-readable prediction.

## How to Run the Code

1. **Clone the Repository:**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```


3.**Ensure Dataset is Present:** Make sure the `sonar.csv` file is located in a sub-folder named `Dataset`.
4.**Run the Jupyter Notebook:** Launch Jupyter Notebook and open the `Sonar.ipynb` file.
    ```bash
    jupyter notebook Sonar.ipynb
    ```
5.Run the cells sequentially to see the entire process from data loading to prediction.

## Conclusion

The Logistic Regression model successfully classifies sonar signals as originating from a rock or a mine with an accuracy of **76.2%** on the test set. This project demonstrates a complete, albeit simple, machine learning pipeline for a binary classification problem. The predictive system showcases how the model can be deployed to make real-world predictions.