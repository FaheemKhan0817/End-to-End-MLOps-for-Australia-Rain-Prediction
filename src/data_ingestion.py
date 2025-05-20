from src.custom_exception import CustomException
from src.logger import get_logger
import os
from config.paths_config import DATA_URL, RAW_DIR

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self):
        # Use the predefined output directory from config
        self.raw_data_path = RAW_DIR
        os.makedirs(self.raw_data_path, exist_ok=True)

    def download_dataset(self):
        try:
            # Import Kaggle API
            from kaggle.api.kaggle_api_extended import KaggleApi

            # Initialize Kaggle API
            api = KaggleApi()
            api.authenticate()

            # Extract dataset details from the URL
            dataset_details = DATA_URL.split('/')
            dataset_name = f"{dataset_details[-2]}/{dataset_details[-1]}"

            # Download the dataset
            api.dataset_download_files(dataset_name, path=self.raw_data_path, unzip=True)

            logger.info(f"Dataset downloaded and saved to {self.raw_data_path}")
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise CustomException(f"Failed to download dataset: {e}")

# Example usage
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.download_dataset()