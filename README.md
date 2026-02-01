 Smart Grid Energy Dispatch & Forecasting System

An AI-powered energy dispatch and forecasting system designed to optimize power distribution in smart grids using machine learning and deep learning models.
The project predicts future energy demand and supports decision-making for adaptive and extreme load conditions.

 Project Overview

Modern smart grids require accurate demand forecasting and intelligent dispatch strategies to ensure efficiency, stability, and cost optimization.
This project implements a hybrid AI-based solution using:

Time-series forecasting

Deep learning (RBI-LSTM)

Machine learning (XGBoost)

Web-based visualization using Flask

The system processes historical energy data, predicts future demand, and generates optimized dispatch results.

 Tech Stack

Programming Language

Python

Frameworks & Libraries

Flask (Web backend)

NumPy, Pandas

Scikit-learn

XGBoost

TensorFlow / Keras

Tools

Git & GitHub

Jupyter Notebook (experimentation)

Joblib (model persistence)

 Project Structure
smart-grid-energy-dispatch/
│
├── backend/
│   ├── app.py                 # Flask application entry point
│   ├── requirements.txt       # Project dependencies
│   │
│   ├── data/
│   │   ├── raw/               # Raw input datasets (ignored in Git)
│   │   ├── processed/         # Cleaned & transformed data
│   │   └── forecasts/         # Model prediction outputs
│   │
│   ├── models/
│   │   ├── rbilstm/            # Deep learning models
│   │   └── xgb/                # XGBoost models
│   │
│   ├── static/                # CSS / JS files
│   └── templates/             # HTML templates
│
├── reports/
│   ├── figures/               # Graphs & visual outputs
│   ├── adaptive/              # Adaptive dispatch results
│   ├── extreme/               # Extreme load case analysis
│   └── opt_results/           # Optimization outputs
│
├── README.md
├── .gitignore

 Features

 Energy demand forecasting using ML & DL models

 Optimized power dispatch for adaptive and extreme conditions

 Flask-based web interface for visualization

 Comparative analysis of forecasting models

 Modular and scalable backend design

 How to Run the Project
1️ Clone the Repository
git clone https://github.com/your-username/smart-grid-energy-dispatch.git
cd smart-grid-energy-dispatch

2️ Install Dependencies
cd backend
pip install -r requirements.txt

3️ Run the Application
python app.py

4️ Open in Browser
http://127.0.0.1:5000/

 Models Used

RBI-LSTM
Captures temporal dependencies for accurate long-term demand forecasting.

XGBoost
Provides fast and interpretable predictions for structured energy data.

 Results

Improved forecasting accuracy compared to baseline methods

Efficient dispatch planning under varying load conditions

Visual analytics for demand trends and optimization outcomes

(Refer to the reports/figures directory for graphs and plots.)

 Notes

Large datasets and trained model files are excluded from GitHub using .gitignore

Models can be retrained using the provided pipeline

The project is structured for easy deployment and future expansion

 Use Cases

Smart grid energy management

Power demand forecasting

Load balancing and dispatch optimization

Academic research & AI-based energy systems

 Author

Krishna Kakade

Engineering Student | AI & ML Enthusiast
