# Simple Linear Regression — Salary Prediction based on Experience
📌 Project Overview

This project implements a Simple Linear Regression model to predict an employee's Salary based on their Years of Experience. It covers the complete workflow — from data cleaning and exploratory data analysis (EDA) to model training and evaluation.

🎯 Problem Statement

Given a dataset containing years of experience and corresponding salary, build a regression model that can predict salary for a given experience level.

📂 Dataset
Features: Experience (years)
Target: Salary
Dataset used: (add source/link here, e.g. Kaggle dataset name)
🛠️ Tech Stack
Python
Pandas — data manipulation
NumPy — numerical operations
Seaborn & Matplotlib — data visualization
Scikit-learn — model building & evaluation
Streamlit - Frontend visualization
🔍 Steps Followed
Data Loading & Cleaning
Handled missing/invalid values
Converted columns to correct numeric types
Exploratory Data Analysis (EDA)
Visualized relationship between Experience and Salary using scatter plots (sns.lmplot)
Checked for outliers and data distribution
Train-Test Split
Split dataset into training and testing sets
Model Training
Trained a Linear Regression model using sklearn.linear_model.LinearRegression
Model Evaluation
Evaluated performance using metrics like R² Score and Mean Squared Error (MSE)
Visualization of Results
Plotted the regression line against actual data points
📊 Results
Metric	Value
R² Score-- 0.98
MSE	-- 0.95



🚀 How to Run
bash
# Clone the repository
git clone https://github.com/mahak-025/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook simple_linear_regression.ipynb
📁 Repository Structure
├── simple_linear_regression.ipynb
├── frontend.py
├──linear_regression_model.pkl
├── dataset.csv
├── requirements.txt
└── README.md
🔮 Future Improvements
Try Multiple Linear Regression with additional features
Compare performance with other regression models (Ridge, Lasso, Polynomial)
Deploy the model as a simple web app
