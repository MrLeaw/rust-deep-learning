from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_breast_cancer
from time import time

timings = {}

now = time()
# run sgd on diabetes dataset
X, y = datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = SGDRegressor()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
ms = time() - now
timings['SGD'] = ms * 1000
print(f"SGDRegressor took {ms}ms")

# run linear regression on diabetes dataset
now = time()
X, y = datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LinearRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
ms = time() - now
timings['Linear'] = ms * 1000
print(f"LinearRegression took {ms}ms")

# Gaussian NB on breast cancer dataset
now = time()
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
ms = time() - now
timings['GaussianNB Breast Cancer'] = ms * 1000
print(f"GaussianNB Breast Cancer took {ms}ms")

# Gaussian NB on iris dataset
now = time()
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
ms = time() - now
timings['GaussianNB Iris'] = ms * 1000
print(f"GaussianNB Iris took {ms}ms")

# Logistic Regression on breast cancer dataset
now = time()
X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
ms = time() - now
timings['Logistic Regression Breast Cancer'] = ms * 1000
print(f"Logistic Regression Breast Cancer took {ms}ms")

# Decision Tree on breast cancer dataset
now = time()
from sklearn.tree import DecisionTreeClassifier
X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
ms = time() - now
timings['Decision Tree Breast Cancer'] = ms * 1000
print(f"Decision Tree Breast Cancer took {ms}ms")

# Decision Tree on iris dataset
now = time()
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
ms = time() - now
timings['Decision Tree Iris'] = ms * 1000
print(f"Decision Tree Iris took {ms}ms")

# write timings to csv
import csv
with open('timings_python.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Model', 'Time (ms)'])
    for key, value in timings.items():
        writer.writerow([key, value])

# now open the `timings.csv` and `timings_python.csv` files and graph the results a bar charts
timings_rust = {}
with open('timings.csv', 'r') as f:
    reader = csv.reader(f)
    # skip the header
    next(reader)
    timings_rust = {rows[0]: float(rows[1]) for rows in reader}

print(timings.values(), timings_rust.values())

# replace Breast Cancer with BC in the keys
timings = {key.replace("Breast Cancer", "BC"): value for key, value in timings.items()}
timings_rust = {key.replace("Breast Cancer", "BC"): value for key, value in timings_rust.items()}

import matplotlib.pyplot as plt
import numpy as np

print("Python timings:", timings)
print("Rust timings:", timings_rust)

fig, axs = plt.subplots(1, len(timings), figsize=(len(timings) * 8, 4))

for i, (key, value) in enumerate(timings.items()):
    ax = axs[i]
    ax.bar(['Python', 'Rust'], [value, timings_rust[key]], color=['blue', 'orange'])
    ax.set_title(f'{key}')
    ax.set_ylabel('Time (ms)')
    ax.set_ylim(0, max(value, timings_rust[key]) * 1.2) 
plt.tight_layout()
plt.subplots_adjust(wspace=0.4)
plt.show()