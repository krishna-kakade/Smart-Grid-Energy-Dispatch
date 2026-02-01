# Smart Grid Energy Dispatch & Forecasting System

An AI-driven smart grid energy dispatch and forecasting system designed to predict electricity demand and support optimized power distribution under adaptive and extreme load conditions using machine learning and deep learning models.

This project focuses on forecast accuracy, system scalability, and clean backend design, making it suitable for academic evaluation and real-world smart grid research.

# Project Overview

Smart grids require accurate demand forecasting and intelligent dispatch strategies to ensure stability, efficiency, and cost optimization.
This project implements a hybrid AI approach combining time-series forecasting, deep learning, and machine learning techniques.

The system:

Processes historical energy data

Predicts future energy demand

Analyzes dispatch strategies for different load scenarios

Visualizes results through a Flask-based backend

# Tech Stack

Programming Language

Python

Frameworks & Libraries

Flask

NumPy, Pandas

Scikit-learn

XGBoost

TensorFlow / Keras

Tools

Git & GitHub

Joblib

Jupyter Notebook

# Project Structure
smart-grid-energy-dispatch/
│
├── backend/
│   ├── app.py                 # Flask application entry point
│   ├── requirements.txt       # Project dependencies
│   ├── data/                  # Data directory (ignored in GitHub)
│   ├── models/                # Trained models (ignored in GitHub)
│   ├── static/                # CSS / JS files
│   └── templates/             # HTML templates
│
├── reports/
│   ├── figures/               # Graphs and visual outputs
│   ├── adaptive/              # Adaptive dispatch results
│   ├── extreme/               # Extreme load case analysis
│   └── opt_results/           # Optimization results
│
├── scripts/                   # Training & preprocessing scripts
│
├── README.md
├── .gitignore

# Features

 Energy demand forecasting using ML and DL models

 Power dispatch analysis for adaptive and extreme scenarios

 Flask-based backend for result visualization

 Model performance and result analysis

 Modular and scalable project structure

# How to Run the Project
1️. Clone the Repository
git clone https://github.com/your-username/smart-grid-energy-dispatch.git
cd smart-grid-energy-dispatch

2️. Install Dependencies
cd backend
pip install -r requirements.txt

3️. Run the Application
python app.py

4️. Open in Browser

# Models Used

RBI-LSTM
Used for capturing long-term temporal dependencies in energy demand data.

XGBoost
Used for fast, accurate predictions on structured energy datasets.

# Results

Improved forecasting accuracy compared to baseline methods

Effective dispatch planning for varying load conditions

Clear visual insights for energy demand and optimization outcomes

Graphs and analysis outputs are available in the reports/ directory.

# Important Note on Data & Models

Due to GitHub file size limitations, the following are intentionally excluded:

Trained model files (.h5, .joblib)

Large datasets (.csv, .xlsx)

These files can be recreated by running the scripts in the scripts/ directory.

This follows industry-standard best practices for version control.

# Use Cases

Smart grid energy management

Electricity demand forecasting

Load balancing and dispatch optimization

Academic research in AI-based energy systems

# Author

Krishna Kakade
Engineering Student | AI & ML Enthusiast
