World Happiness Classification - Advanced Machine Learning Project

Project Overview

This project is part of an advanced machine learning course, aiming to classify countries based on happiness scores using World Happiness Report 2023 data. The goal is to apply various machine learning techniques, improve model accuracy, and document the entire process.

Team Members

Yizhao Hong

Jason Lee

Rick Hua

Repository Structure

📂 Advanced-Machine-Learning-World-Happiness-Classification
│── 📄 README.md           # Project documentation
│── 📄 requirements.txt     # Required Python dependencies
│── 📂 data                # Raw and processed datasets
│── 📂 notebooks           # Jupyter notebooks for analysis & modeling
│── 📂 models              # Saved trained models
│── 📂 src                 # Python scripts for data preprocessing & modeling
│── 📂 results             # Evaluation results & visualizations
│── 📄 report.pdf          # Final report

Project Objectives

✔ Understand model functionalities and hyperparameters
✔ Perform data preprocessing and feature engineering
✔ Train various classification models (Random Forest, Gradient Boosting, Deep Learning)
✔ Optimize models using GridSearchCV
✔ Evaluate model performance using accuracy metrics

Data Processing & Feature Engineering
Dataset
World Happiness Report 2023
Additional socioeconomic data

Key Preprocessing Steps
✅ Handle missing values using SimpleImputer
✅ Normalize numerical features with StandardScaler
✅ Convert categorical variables using OneHotEncoder
✅ Create interaction features (poverty ratio, education gap, etc.)

Machine Learning Models
Baseline Model: Random Forest
RandomForestClassifier(n_estimators=100, random_state=42)
Accuracy: 57.1%
Optimized Model: Random Forest (Tuned)
n_estimators=500, max_depth=10, max_features='log2'
Accuracy: 64.3%
Hyperparameter Optimization using GridSearchCV
Tested multiple values for max_depth, n_estimators, min_samples_split, etc.
Best model accuracy: 58.9%
Gradient Boosting Model
GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.1)
Accuracy: 50%
Deep Learning Model
Neural Network (Keras & TensorFlow)
Architecture:
4 Hidden Layers with ReLU activation
Output layer with Softmax activation for multi-class classification
Training Details:
loss='categorical_crossentropy'
optimizer='sgd' (Stochastic Gradient Descent)
Trained for 300 epochs
Key Insights & Learnings
🔹 Feature Engineering significantly impacts model performance
🔹 Hyperparameter Tuning improved accuracy from 57.1% → 64.3%
🔹 Deep Learning overfitting – model achieved 100% accuracy in training but performed poorly in validation
🔹 Using Adam Optimizer instead of SGD could improve deep learning training efficiency


