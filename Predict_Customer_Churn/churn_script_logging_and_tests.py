"""
This module contains test functions for churn_library.py.

It performs unit tests on all functions in the churn_library module
and logs success/failure results to a .log file.

Author: Your Name
Date: 2024-02-20
"""

import os
import logging
import churn_library as cls  # Changed from churn_library_solution to churn_library

# Configure logging
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('churn_library_tests')

def test_import(import_data):
    '''
    Test data import functionality
    
    Args:
        import_data: Function to test
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logger.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logger.info("Testing import_data: DataFrame has rows and columns")
    except AssertionError as err:
        logger.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err

def test_eda(perform_eda):
    '''
    Test perform eda functionality
    
    Args:
        perform_eda: Function to test
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        perform_eda(df)
        
        # Check if EDA images were created
        eda_images = [
            'churn_distribution.png',
            'correlation_heatmap.png'
        ]
        
        for image in eda_images:
            assert os.path.exists(f"./images/eda/{image}")
        logger.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logger.error("Testing perform_eda: Not all EDA images were created")
        raise err
    except Exception as err:
        logger.error("Testing perform_eda: Error occurred while performing EDA")
        raise err

def test_encoder_helper(encoder_helper):
    '''
    Test encoder helper functionality
    
    Args:
        encoder_helper: Function to test
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        
        category_lst = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]
        
        df_encoded = encoder_helper(df, category_lst, 'Churn')
        
        # Check if new columns were created
        for category in category_lst:
            assert f"{category}_Churn" in df_encoded.columns
        
        logger.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logger.error("Testing encoder_helper: Not all categorical columns were encoded")
        raise err
    except Exception as err:
        logger.error("Testing encoder_helper: Error occurred during encoding")
        raise err

def test_perform_feature_engineering(perform_feature_engineering):
    '''
    Test perform_feature_engineering functionality
    
    Args:
        perform_feature_engineering: Function to test
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        
        # Prepare data
        category_lst = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]
        df = cls.encoder_helper(df, category_lst, 'Churn')
        
        # Test feature engineering
        X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')
        
        # Check if data splits exist and have correct shapes
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) > 0
        assert len(X_test) > 0
        
        logger.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logger.error("Testing perform_feature_engineering: Data split failed")
        raise err
    except Exception as err:
        logger.error("Testing perform_feature_engineering: Error occurred during feature engineering")
        raise err

def test_train_models(train_models):
    '''
    Test train_models functionality
    
    Args:
        train_models: Function to test
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        
        # Prepare data
        category_lst = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]
        df = cls.encoder_helper(df, category_lst, 'Churn')
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df, 'Churn')
        
        # Test model training
        train_models(X_train, X_test, y_train, y_test)
        
        # Check if models were saved
        assert os.path.exists("./models/rfc_model.pkl")
        assert os.path.exists("./models/logistic_model.pkl")
        
        # Check if result images were created
        result_images = [
            'roc_curves.png',
            'rf_feature_importance.png',
            'logistic_report.png',
            'rf_report.png'
        ]
        
        for image in result_images:
            assert os.path.exists(f"./images/results/{image}")
        
        logger.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logger.error("Testing train_models: Not all models or result images were created")
        raise err
    except Exception as err:
        logger.error("Testing train_models: Error occurred during model training")
        raise err

if __name__ == "__main__":
    logger.info("Starting tests...")
    
    # Run all tests
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
    
    logger.info("All tests completed successfully!")