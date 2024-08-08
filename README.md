# Evaluate execution of Back Exercise using Body Sensor and ML

## Project Overview

This project is focused on developing a machine learning classifier to accurately evaluate the correctness of back exercises performed by individuals, monitored through body sensors. The primary dataset includes sensor data from individuals performing three distinct types of back exercises, which are labeled as correctly or incorrectly executed based on predefined standards.

## Dataset Description

The dataset features sensor readings from 7 individuals, collected through sensors attached to their backs as they performed various exercises. The sensor data includes measurements such as acceleration, gyroscope values, angles, and other dynamic metrics, spread across 70 different columns. This comprehensive data allows us to analyze and predict the exercise execution quality.

## Goal

The objective of this project is to create a classifier that can predict whether back exercises are performed correctly. This classifier will help in coaching, rehabilitation, and personal training by ensuring exercises are done properly to avoid injuries.

## Test Scenario

The classifier will be particularly tested on the performance of an older man to ensure its effectiveness across different age groups. In the test scenario, the model achieved a perfect score by correctly labeling all 9 exercises performed by this individual, demonstrating its potential accuracy and reliability in real-world applications.

## Machine Learning Workflow

Data Preprocessing: Not needed because the data provided were already transformed
Feature Engineering: Also not needed, the original dataset was ready to use, also since we had too many variable a Lasso Feature Selection was already applied, bringing the number of columns to a reasonable number.
Model Selection: Exploring various classification algorithms, starting by Boosting model (including hyperparameters optimization), we overestimated the problem. In the end a Random Forest model was enough and we earned in interpretability.
Model Training and Validation: Employing techniques like cross-validation and Optuna to train and fine-tune the models, ensuring they generalize well on unseen data.
Evaluation: Accurancy
