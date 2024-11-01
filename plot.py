# load y_test.csv and y_pred.csv (they contain one value per line)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create charts directory if it doesn't exist
if not os.path.exists('charts'):
    os.makedirs('charts')

# clear the directory
for file in os.listdir('charts'):
    os.remove(f'charts/{file}')

SETS = [("Gradient Descent Classifier", "y_test.csv", "y_pred.csv"), ("Linear Regression Classifier", "ylr_test.csv", "ylr_pred.csv")]

for SET in SETS:
    y_test = pd.read_csv(SET[1])
    # get the only column as an array
    y_test = y_test.iloc[:, 0].values
    predictions = pd.read_csv(SET[2])
    # get the only column as an array
    predictions = predictions.iloc[:, 0].values
    
    # Plot the results
    # Ideally, the points should be close to a diagonal line
    # graph the diagonal line to see how far the points are from the line
    plt.scatter(y_test, predictions)
    plt.plot([0, 330], [0, 330], color='red')
    # also draw a line for the predicted values
    trendline = np.polyfit(y_test, predictions, 1)
    trendline = np.poly1d(trendline)
    
    plt.plot(y_test, trendline(y_test), color='green')
    
    plt.xlabel("Actual Target")
    plt.ylabel("Predicted Target")
    plt.title(SET[0])
    
    # Save the plot as an image file
    plt.savefig(f'charts/{SET[0].replace(" ", "_")}.png')
    plt.close()
 

matrices = {
    "Naive Bayes (Breast Cancer)": "matrix.csv",
    "Naive Bayes (Iris)": "matrix_iris.csv",
    "Logistic Regression (Breast Cancer)": "matrix_logistic_regression.csv",
    "Decision Tree (Breast Cancer)": "matrix_decision_tree.csv",
    "Decision Tree (Iris)": "matrix_decision_tree_iris.csv"
}

for key, value in matrices.items():
    matrix = []
    classes = []
    with open(value, "r", encoding='utf-8') as f:
        classes = f.readline().split(",")
        for line in f:
            matrix.append([int(x) for x in line.split(",")])

    # Plot the confusion matrix
    plt.matshow(matrix)
    # Add the values to the plot
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            plt.text(j, i, str(matrix[i][j]), ha='center', va='center')
    plt.title(f"{key}")
    plt.colorbar()
    plt.savefig(f'charts/{key.replace(" ", "_")}_confusion_matrix.png')
    plt.close()
