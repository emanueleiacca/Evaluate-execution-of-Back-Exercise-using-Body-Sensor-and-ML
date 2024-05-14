# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:04:22 2024

@author: feder
"""

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

file_path = "AF3_FeatReduc.xlsx"
df_Sistine = pd.read_excel(file_path)
y=df_Sistine["Type"]
X= df_Sistine.iloc[:, 2:-1]
subjects=df_Sistine["Sub"]

## Modello 1. senza cross validation accuracy 98%
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #split 80-20
# rf_classifier.fit(X_train, y_train) # Addestrare il modello sul set di addestramento
# accuracy = rf_classifier.score(X_test, y_test) # Valutare il modello sul set di test
# print("Accuracy:", accuracy)

# Definisci il modello
model = RandomForestClassifier(n_estimators=6000)
# Inizializza l'oggetto Leave-One-Group-Out Cross-Validation (LOGOCV)
logo = LeaveOneGroupOut()
# Inizializza una lista per memorizzare i punteggi di accuratezza
scores = []
# Esegui la Leave-One-Group-Out Cross-Validation
for train_index, test_index in logo.split(X, y, subjects):
    # Dividi i dati in set di addestramento e test per il fold corrente
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    # **Normalizzare X train**#
   
    
    # Calculate z-score
    mean = np.mean(X_train)
    std_dev = np.std(X_train)
    X_train_n, MU, SIGMA = (X_train-X_train.mean())/X_train.std(), X_train.mean(), X_train.std() 
    X_test_n= (X_test - MU) / SIGMA
    
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # Addestra il modello sul set di addestramento
    model.fit(X_train_n, y_train)
    # Esegui le predizioni sul set di test
    y_pred = model.predict(X_test_n)
    # Calcola l'accuratezza sul set di test e aggiungila alla lista dei punteggi
    accuracy = accuracy_score(y_test, y_pred)
    scores.append(accuracy)
# Calcola l'accuratezza media su tutti i gruppi
mean_accuracy = sum(scores) / len(scores)
print("Mean Accuracy:", mean_accuracy)
print("Scores:",scores)
print(np.mean(MU))