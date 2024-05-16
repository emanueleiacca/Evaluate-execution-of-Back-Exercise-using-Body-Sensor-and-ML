# -*- coding: utf-8 -*-
"""
Created on Wed May 15 09:37:44 2024

@author: ele99
"""


from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


# Carica i dati di addestramento
train_file_path = r"AF3_FeatReduc.xlsx"
df_train = pd.read_excel(train_file_path)

# Estrai le variabili indipendenti (X) e la variabile dipendente (y) dai dati di addestramento
X_train = df_train.iloc[:, 2:-1]
y_train = df_train["Type"]
subjects_train = df_train["Sub"]

# Carica i dati di test (un singolo soggetto)
test_file_path = r"AF3_A016.xlsx"
df_test = pd.read_excel(test_file_path)

# Estrai le variabili indipendenti (X) e la variabile dipendente (y) dai dati di test
X_test = df_test.iloc[:,:-1]
y_test = df_test["Type"]

# Definisci il modello
model = RandomForestClassifier(n_estimators=30)
# Inizializza l'oggetto Leave-One-Group-Out Cross-Validation (LOGOCV)
logo = LeaveOneGroupOut()
# Inizializza una lista per memorizzare i punteggi di accuratezza
scores = []
# Esegui la Leave-One-Group-Out Cross-Validation
for train_index, test_index in logo.split(X_train, y_test, subjects_train):
    # Dividi i dati in set di addestramento e test per il fold corrente
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
   
    mean = np.mean(X_train_fold)
    std_dev = np.std(X_train_fold)
    X_train_fold_n, MU, SIGMA = (X_train_fold-X_train_fold.mean())/X_train_fold.std(), X_train_fold.mean(), X_train_fold.std() 
    # Normalizza X train
   
    X_test_fold_n = (X_test_fold - MU) / SIGMA

    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    # Addestra il modello sul set di addestramento
    model.fit(X_train_fold_n, y_train_fold)
    # Esegui le predizioni sul set di test
    y_pred = model.predict(X_test_fold_n)
    # Calcola l'accuratezza sul set di test e aggiungila alla lista dei punteggi
    accuracy = accuracy_score(y_test_fold, y_pred)
    scores.append(accuracy)
# Calcola l'accuratezza media su tutti i gruppi
mean_accuracy = sum(scores) / len(scores)
print("Mean Accuracy:", mean_accuracy)
print("Scores:", scores)
print(np.mean(MU))
