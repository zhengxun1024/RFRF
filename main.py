import numpy as np
import pandas as pd
from RSFRF.rsfrf import RandomForestClassifier, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    filename = 'data/small/iris.csv'
    df = pd.read_csv(filename, header=None)
    data = df.values
    X, y = data[:, :-1], data[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    print(f'Accuracy: {acc:.4f}')
