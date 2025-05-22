import pandas as pd
import numpy as np
import pickle
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_recall_curve
import joblib
from src.custom_exception import CustomException
from src.logger import get_logger
from config.paths_config import PROCESSED_DIR, MODEL_DIR

class ModelTrainer:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.best_model = None
        self.top_feature_names = [
            'RainToday', 'Cloud3pm_missing', 'Sunshine', 'Humidity3pm',
            'Cloud3pm', 'Sunshine_missing', 'Cloud9am_missing', 'Rainfall',
            'WindGustDiff', 'Cloud9am', 'Pressure3pm', 'CloudCoverAvg'
        ]
        self.threshold = None

    def hypertune_xgboost(self, X_train_smote, y_train_smote, X_test, y_test):
        try:
            self.logger.info("Hypertuning XGBoost with RandomizedSearchCV")
            # Subset to top 12 features
            X_train_smote_top12 = X_train_smote[self.top_feature_names]
            X_test_top12 = X_test[self.top_feature_names]

            xgb_param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'scale_pos_weight': [len(y_train_smote[y_train_smote==0])/len(y_train_smote[y_train_smote==1]), 1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }

            xgb_random_search = RandomizedSearchCV(
                estimator=XGBClassifier(random_state=42),
                param_distributions=xgb_param_grid,
                n_iter=20,
                cv=5,
                scoring='recall_macro',
                n_jobs=-1,
                verbose=2,
                random_state=42
            )
            xgb_random_search.fit(X_train_smote_top12, y_train_smote)

            self.best_model = xgb_random_search.best_estimator_
            y_pred_best_xgb = self.best_model.predict(X_test_top12)
            accuracy = accuracy_score(y_test, y_pred_best_xgb)
            self.logger.info(f"Best Parameters for XGBoost: {xgb_random_search.best_params_}")
            self.logger.info(f"Accuracy after Hypertuning with Top 12 Features: {accuracy}")
            self.logger.info(f"Classification Report after Hypertuning:\n{classification_report(y_test, y_pred_best_xgb)}")
            self.logger.info(f"Confusion Matrix after Hypertuning:\n{confusion_matrix(y_test, y_pred_best_xgb)}")

            return X_train_smote_top12, X_test_top12
        except Exception as e:
            self.logger.error(f"Error in hypertuning XGBoost: {str(e)}")
            raise CustomException("Failed to hypertune XGBoost", str(e))

    def tune_threshold(self, X_test_top12, y_test, target_recall=0.75):
        try:
            self.logger.info(f"Tuning threshold to achieve recall >= {target_recall}")
            y_scores = self.best_model.predict_proba(X_test_top12)[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
            valid_indices = np.where(recall >= target_recall)[0]

            if len(valid_indices) == 0:
                self.logger.warning(f"No threshold achieves recall >= {target_recall}. Using the highest recall available.")
                valid_indices = [np.argmax(recall)]

            best_index = valid_indices[np.argmax(precision[valid_indices])]
            self.threshold = thresholds[best_index]
            y_pred_adjusted = (y_scores >= self.threshold).astype(int)

            accuracy = accuracy_score(y_test, y_pred_adjusted)
            self.logger.info(f"Accuracy with Adjusted Threshold (Recall >= {target_recall}): {accuracy}")
            self.logger.info(f"Classification Report with Adjusted Threshold:\n{classification_report(y_test, y_pred_adjusted)}")
            self.logger.info(f"Confusion Matrix with Adjusted Threshold:\n{confusion_matrix(y_test, y_pred_adjusted)}")
            self.logger.info(f"Selected Threshold: {self.threshold:.3f}")
            self.logger.info(f"Precision at Selected Threshold: {precision[best_index]:.3f}")
            self.logger.info(f"Recall at Selected Threshold: {recall[best_index]:.3f}")

        except Exception as e:
            self.logger.error(f"Error in threshold tuning: {str(e)}")
            raise CustomException("Failed to tune threshold", str(e))

    def save_model(self, model_filename='xgboost_rain_prediction_model.pkl', threshold_filename='threshold.txt'):
        try:
            # Create model directory if it doesn't exist
            os.makedirs(MODEL_DIR, exist_ok=True)
            model_path = os.path.join(MODEL_DIR, model_filename)
            threshold_path = os.path.join(MODEL_DIR, threshold_filename)

            self.logger.info(f"Saving model to {model_path}")
            joblib.dump(self.best_model, model_path)
            self.logger.info(f"Saving threshold to {threshold_path}")
            with open(threshold_path, 'w') as f:
                f.write(str(self.threshold))
            self.logger.info("Model and threshold saved successfully")
        except Exception as e:
            self.logger.error(f"Error in saving model: {str(e)}")
            raise CustomException("Failed to save model", str(e))

if __name__ == "__main__":
    # Load processed data
    x_train_path = os.path.join(PROCESSED_DIR, "X_train.pkl")
    y_train_path = os.path.join(PROCESSED_DIR, "y_train.pkl")
    x_test_path = os.path.join(PROCESSED_DIR, "X_test.pkl")
    y_test_path = os.path.join(PROCESSED_DIR, "y_test.pkl")
    x_train_smote_path = os.path.join(PROCESSED_DIR, "X_train_smote.pkl")
    y_train_smote_path = os.path.join(PROCESSED_DIR, "y_train_smote.pkl")

    try:
        with open(x_train_path, 'rb') as f:
            X_train = pickle.load(f)
        with open(y_train_path, 'rb') as f:
            y_train = pickle.load(f)
        with open(x_test_path, 'rb') as f:
            X_test = pickle.load(f)
        with open(y_test_path, 'rb') as f:
            y_test = pickle.load(f)
        with open(x_train_smote_path, 'rb') as f:
            X_train_smote = pickle.load(f)
        with open(y_train_smote_path, 'rb') as f:
            y_train_smote = pickle.load(f)
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error loading processed data: {str(e)}")
        raise CustomException("Failed to load processed data", str(e))

    # Initialize model trainer
    model_trainer = ModelTrainer()

    # Hypertune XGBoost with top 12 features
    X_train_smote_top12, X_test_top12 = model_trainer.hypertune_xgboost(X_train_smote, y_train_smote, X_test, y_test)

    # Tune threshold
    model_trainer.tune_threshold(X_test_top12, y_test, target_recall=0.75)

    # Save the model
    model_trainer.save_model()