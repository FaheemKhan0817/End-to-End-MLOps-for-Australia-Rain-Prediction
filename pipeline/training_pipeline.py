from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessor
from src.model_training import ModelTrainer
from src.logger import get_logger

if __name__ == "__main__":
    # Initialize logger
    logger = get_logger(__name__)
    logger.info("Starting the training pipeline")

    try:
        # Step 1: Data Ingestion
        logger.info("Initiating data ingestion")
        data_ingestion = DataIngestion()
        data_ingestion.download_dataset()
        logger.info("Data ingestion completed")

        # Step 2: Data Processing
        logger.info("Initiating data processing")
        data_processor = DataProcessor()
        data_processor.load_data()
        data_processor.check_duplicates()
        data_processor.preprocess_data()
        data_processor.encode_data()
        data_processor.split_data()
        data_processor.apply_smote()
        data_processor.save_processed_data()
        X_train, X_test, y_train, y_test, X_train_smote, y_train_smote = data_processor.get_processed_data()
        logger.info(f"Data processing completed. Processed data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}, X_train_smote: {X_train_smote.shape}")

        # Step 3: Model Training
        logger.info("Initiating model training")
        model_trainer = ModelTrainer()
        X_train_smote_top12, X_test_top12 = model_trainer.hypertune_xgboost(X_train_smote, y_train_smote, X_test, y_test)
        model_trainer.tune_threshold(X_test_top12, y_test, target_recall=0.75)
        model_trainer.save_model()
        logger.info("Model training completed")

        logger.info("Training pipeline completed successfully")

    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise