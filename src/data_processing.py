import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from src.custom_exception import CustomException
from src.logger import get_logger
from config.paths_config import RAW_DIR, PROCESSED_DIR

class DataProcessor:
    def __init__(self, data_path=None):
        # If no data_path is provided, default to RAW_DIR/weatherAUS.csv
        self.data_path = data_path if data_path else os.path.join(RAW_DIR, "weatherAUS.csv")
        self.logger = get_logger(__name__)
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_smote = None
        self.y_train_smote = None

    def load_data(self):
        try:
            self.logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            self.logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise CustomException("Failed to load data", str(e))

    def check_duplicates(self):
        try:
            self.logger.info("Checking for duplicates")
            duplicates = self.df.duplicated().sum()
            self.logger.info(f"Number of duplicates found: {duplicates}")
            return duplicates
        except Exception as e:
            self.logger.error(f"Error checking duplicates: {str(e)}")
            raise CustomException("Failed to check duplicates", str(e))

    def preprocess_data(self):
        try:
            self.logger.info("Starting data preprocessing")

            # Drop rows where target variable is missing
            self.df = self.df.dropna(subset=['RainTomorrow'])
            self.logger.info(f"Shape after dropping missing RainTomorrow: {self.df.shape}")

            # Convert Date to datetime and extract features
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df['Year'] = self.df['Date'].dt.year
            self.df['Month'] = self.df['Date'].dt.month
            self.df['Day'] = self.df['Date'].dt.day
            self.df = self.df.drop('Date', axis=1)

            # Handle missing values for numerical columns
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                self.df[col] = self.df[col].fillna(self.df[col].median())

            # Handle missing values for categorical columns
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

            # Feature Engineering
            self.df['TempDiff'] = self.df['MaxTemp'] - self.df['MinTemp']
            self.df['PressureDiff'] = self.df['Pressure9am'] - self.df['Pressure3pm']
            self.df['WindGustDiff'] = self.df['WindGustSpeed'] - self.df['WindSpeed3pm']
            self.df['CloudCoverAvg'] = (self.df['Cloud9am'] + self.df['Cloud3pm']) / 2

            # Add missing value indicators
            for col in ['Sunshine', 'Cloud9am', 'Cloud3pm']:
                self.df[f'{col}_missing'] = self.df[col].isnull().astype(int)

            self.logger.info("Data preprocessing completed")
        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {str(e)}")
            raise CustomException("Failed to preprocess data", str(e))

    def encode_data(self):
        try:
            self.logger.info("Encoding categorical variables")
            le = LabelEncoder()
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col != 'RainTomorrow':
                    self.df[col] = le.fit_transform(self.df[col])
            # Encode target variable
            self.df['RainTomorrow'] = le.fit_transform(self.df['RainTomorrow'])
            self.logger.info("Encoding completed")
        except Exception as e:
            self.logger.error(f"Error in encoding data: {str(e)}")
            raise CustomException("Failed to encode data", str(e))

    def split_data(self, test_size=0.2, random_state=42):
        try:
            self.logger.info("Splitting data into train and test sets")
            self.X = self.df.drop('RainTomorrow', axis=1)
            self.y = self.df['RainTomorrow']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
            )
            self.logger.info(f"Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
        except Exception as e:
            self.logger.error(f"Error in splitting data: {str(e)}")
            raise CustomException("Failed to split data", str(e))

    def apply_smote(self, random_state=42):
        try:
            self.logger.info("Applying SMOTE to balance training data")
            smote = SMOTE(random_state=random_state)
            self.X_train_smote, self.y_train_smote = smote.fit_resample(self.X_train, self.y_train)
            self.logger.info(f"Shape after SMOTE - X_train_smote: {self.X_train_smote.shape}, y_train_smote: {self.y_train_smote.shape}")
        except Exception as e:
            self.logger.error(f"Error in applying SMOTE: {str(e)}")
            raise CustomException("Failed to apply SMOTE", str(e))

    def save_processed_data(self):
        try:
            # Create processed directory if it doesn't exist
            os.makedirs(PROCESSED_DIR, exist_ok=True)
            self.logger.info("Saving processed data in pickle format")
            # Define file paths
            x_train_path = os.path.join(PROCESSED_DIR, "X_train.pkl")
            y_train_path = os.path.join(PROCESSED_DIR, "y_train.pkl")
            x_test_path = os.path.join(PROCESSED_DIR, "X_test.pkl")
            y_test_path = os.path.join(PROCESSED_DIR, "y_test.pkl")
            x_train_smote_path = os.path.join(PROCESSED_DIR, "X_train_smote.pkl")
            y_train_smote_path = os.path.join(PROCESSED_DIR, "y_train_smote.pkl")

            # Save data
            with open(x_train_path, 'wb') as f:
                pickle.dump(self.X_train, f)
            with open(y_train_path, 'wb') as f:
                pickle.dump(self.y_train, f)
            with open(x_test_path, 'wb') as f:
                pickle.dump(self.X_test, f)
            with open(y_test_path, 'wb') as f:
                pickle.dump(self.y_test, f)
            with open(x_train_smote_path, 'wb') as f:
                pickle.dump(self.X_train_smote, f)
            with open(y_train_smote_path, 'wb') as f:
                pickle.dump(self.y_train_smote, f)

            self.logger.info(f"Processed data saved to {PROCESSED_DIR}")
        except Exception as e:
            self.logger.error(f"Error in saving processed data: {str(e)}")
            raise CustomException("Failed to save processed data", str(e))

    def get_processed_data(self):
        return (self.X_train, self.X_test, self.y_train, self.y_test,
                self.X_train_smote, self.y_train_smote)

if __name__ == "__main__":
    # Example usage of DataProcessor
    data_processor = DataProcessor()
    data_processor.load_data()
    data_processor.check_duplicates()
    data_processor.preprocess_data()
    data_processor.encode_data()
    data_processor.split_data()
    data_processor.apply_smote()
    data_processor.save_processed_data()
    X_train, X_test, y_train, y_test, X_train_smote, y_train_smote = data_processor.get_processed_data()
    print(f"Processed data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}, X_train_smote: {X_train_smote.shape}")