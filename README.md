🌦️ End-to-End MLOps for Australia Rain Prediction
Welcome to the End-to-End MLOps for Australia Rain Prediction project! This project showcases a complete machine learning operations (MLOps) pipeline for predicting rainfall in Australia using the WeatherAUS dataset. From data ingestion to model deployment, it demonstrates best practices in MLOps, integrating modern tools and automation for a seamless workflow.

📋 Table of Contents

Project Overview
Features
Tech Stack
Project Structure
Installation
Usage
Training the Model
CI/CD Pipeline with GitLab
Deployment on Render
Contributing
License
Acknowledgements


🌟 Project Overview
This project leverages the WeatherAUS dataset to predict whether it will rain tomorrow across various Australian locations. The pipeline encompasses:

Data Ingestion: Automatically fetches and processes the dataset.
Preprocessing: Handles missing values, encodes categorical variables, and applies SMOTE for imbalanced data.
Model Training: Trains an XGBoost model with hyperparameter tuning and threshold optimization.
Web Application: Deploys a Flask-based web app for real-time rain predictions.
Automation: Uses GitLab CI/CD for continuous integration and deployment.
Hosting: Deploys the app on Render’s free tier using Docker.

The result is a scalable, user-friendly application for weather predictions, backed by a robust MLOps pipeline.

✨ Features

Automated Data Ingestion: Downloads and prepares the WeatherAUS dataset.
Advanced Preprocessing: Handles missing data, feature encoding, and class imbalance with SMOTE.
Optimized Model: Trains an XGBoost model with fine-tuned hyperparameters and an optimized decision threshold.
Interactive Web Interface: A Flask app with a form for users to input weather features and receive predictions.
CI/CD Automation: Automates building, testing, and deployment using GitLab CI/CD.
Cloud Deployment: Hosts the app on Render with a free-tier plan for scalability.


🛠️ Tech Stack



Category
Tools



Programming Language
Python 3.9


Machine Learning
XGBoost, Scikit-learn, Imbalanced-learn


Web Framework
Flask


Containerization
Docker


CI/CD
GitLab CI/CD


Hosting
Render



📂 Project Structure
End-to-End-MLOps-for-Australia-Rain-Prediction/
├── app.py                      # Flask app for serving predictions
├── src/
│   ├── data_ingestion.py       # Downloads the WeatherAUS dataset
│   ├── data_processing.py      # Preprocessing and feature engineering
│   ├── model_training.py       # Trains the XGBoost model
│   ├── training_pipeline.py    # Orchestrates the training pipeline
│   ├── custom_exception.py     # Custom exception handling
│   ├── logger.py               # Logging utility
│   └── __init__.py
├── config/
│   ├── paths_config.py         # Path configurations
│   └── __init__.py
├── static/
│   ├── css/
│   │   └── styles.css          # CSS for the web interface
│   └── js/
│       └── script.js           # JavaScript for the web interface
├── templates/
│   └── index.html              # HTML template for the Flask app
├── artifacts/
│   ├── model/
│   │   ├── xgboost_rain_prediction_model.pkl  # Trained model
│   │   └── threshold.txt       # Optimized threshold
│   ├── raw/                    # Raw dataset (not in GitLab)
│   └── processed/              # Processed data (not in GitLab)
├── Dockerfile                  # Docker configuration
├── .gitlab-ci.yml              # GitLab CI/CD pipeline configuration
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation


🚀 Installation
Prerequisites

Python 3.9+
Docker
GitLab repository access
Render account (free tier)

Steps

Clone the Repository:
git clone https://gitlab.com/faheemkhan08171/end-to-end-mlops-for-australia-rain-prediction.git
cd end-to-end-mlops-for-australia-rain-prediction


Install Dependencies:
pip install -r requirements.txt


Set Up Environment:

Ensure artifacts/model/ contains xgboost_rain_prediction_model.pkl and threshold.txt (included in the repository).




🖥️ Usage
Running Locally

Start the Flask App:
python app.py


Access the Web Interface:

Open http://localhost:5000 in your browser.
Fill in the weather features (e.g., RainToday, Humidity3pm).
Submit to get a prediction (e.g., "Yes, it will rain tomorrow! Probability: 65.43%").



Running with Docker

Build the Docker Image:
docker build -t mlops-app .


Run the Container:
docker run -p 5000:5000 mlops-app


Access the App:

Visit http://localhost:5000 in your browser.




🧠 Training the Model
To retrain the model with updated data:

Run the Training Pipeline:
python src/training_pipeline.py


This downloads the WeatherAUS dataset, preprocesses it, applies SMOTE, trains the XGBoost model, and saves artifacts to artifacts/model/.


Verify Artifacts:

Check that xgboost_rain_prediction_model.pkl and threshold.txt are updated in artifacts/model/.




Note: Processed data (X_train.pkl, etc.) is not stored in the repository due to its size (~130 MB) but is generated during training.


🔄 CI/CD Pipeline with GitLab
The project uses GitLab CI/CD to automate building, pushing, and deploying the Dockerized app to Render.
Workflow Overview

Trigger: Runs on pushes to the main branch.
Steps:
Builds a Docker image using the Dockerfile.
Pushes the image to a container registry (e.g., Docker Hub).
Deploys the image to Render using Render’s API.



Setup

Container Registry:

Create a Docker Hub account.
Add the following variables in GitLab CI/CD settings (Project → Settings → CI/CD → Variables):
DOCKERHUB_USERNAME: Your Docker Hub username.
DOCKERHUB_PASSWORD: Your Docker Hub password or access token.




Render API Key:

Sign up for a Render account and generate an API key (Render Dashboard → Account Settings → API Keys).
Add the following variables in GitLab CI/CD settings:
RENDER_API_KEY: Your Render API key.
RENDER_SERVICE_ID: Your Render service ID (found in the Render Dashboard).




Pipeline Configuration:

The pipeline is defined in .gitlab-ci.yml. It includes stages for checkout, build, and deployment to Kubernetes (or Render in this case).




☁️ Deployment on Render
The app is hosted on Render’s free tier for scalability and ease of use.
Deployment Steps

Create a Render Account:

Sign up at Render.


Set Up a Web Service:

In the Render Dashboard, create a new Web Service.
Select Docker as the runtime.
Link your Docker Hub repository (e.g., yourusername/mlops-app).
Configure:
Port: 5000
Environment Variables:
FLASK_APP: app.py
FLASK_ENV: production


Use the free tier plan.




Deploy Automatically via CI/CD:

The GitLab CI/CD pipeline deploys the app to Render after pushing the Docker image to Docker Hub.
Alternatively, trigger a manual deploy in the Render Dashboard.


Access the App:

https://end-to-end-mlops-for-australia-rain.onrender.com/
Use the form on the app to make predictions.




🤝 Contributing
Contributions are welcome! To contribute:

Fork the repository on GitLab.
Create a new branch (git checkout -b feature/your-feature).
Make changes and commit (git commit -m "Add your feature").
Push to your branch (git push origin feature/your-feature).
Open a Merge Request on GitLab.

Please ensure your code follows the project’s style and includes tests if applicable.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

🙏 Acknowledgements

WeatherAUS Dataset: For providing the data used in this project.
Open-Source Community: For tools like Flask, XGBoost, and Docker.
Render: For offering a free tier to host the application.


Happy Predicting! 🌦️
