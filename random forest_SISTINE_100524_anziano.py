from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Training
train_file_path = r"AF3_FeatReduc.xlsx"
df_train = pd.read_excel(train_file_path)

# Estrai X e y da Train
X_train = df_train.iloc[:, 2:-1]
y_train = df_train["Type"]
subjects_train = df_train["Sub"]

# Test
test_file_path = r"AF3_A016.xlsx"
df_test = pd.read_excel(test_file_path)

# Estrai X e y da Test
X_test = df_test.iloc[:, :-1]
y_test = df_test["Type"]

# modello
model = RandomForestClassifier(n_estimators=30)

# LOGOCV ma usare GroupCv credo sia piu' efficiente
logo = LeaveOneGroupOut()

# lista vuota dove vanno punteggi
scores = []

# CV
for train_index, test_index in logo.split(X_train, y_train, groups=subjects_train):
    # Split CV
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    
    # Normalizza dati
    MU = X_train_fold.mean()
    SIGMA = X_train_fold.std()
    X_train_fold_n = (X_train_fold - MU) / SIGMA
    X_test_fold_n = (X_test_fold - MU) / SIGMA

    # Train 
    model.fit(X_train_fold_n, y_train_fold)
    
    # predictions su train
    y_pred_fold = model.predict(X_test_fold_n)
    
    # accuracy sul train
    accuracy = accuracy_score(y_test_fold, y_pred_fold)
    scores.append(accuracy)

# ean accuracy 
mean_accuracy = np.mean(scores)
print("Mean Accuracy from LOGO-CV:", mean_accuracy)
print("Individual Scores from LOGO-CV:", scores)

# Normalizza il train set
MU = X_train.mean()
SIGMA = X_train.std()
X_train_n = (X_train - MU) / SIGMA

# Train 
model.fit(X_train_n, y_train)

# Normalizza anche il test con stesso metodo
X_test_n = (X_test - MU) / SIGMA

# predictions su test
y_pred_test = model.predict(X_test_n)

#  accuracy su test 
test_accuracy = accuracy_score(y_test, y_pred_test)
print("Accuracy on test data:", test_accuracy)
