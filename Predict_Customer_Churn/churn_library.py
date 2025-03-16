"""
This module contains functions for predicting customer churn.

The module provides functionality for:
- Data import and validation
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training and evaluation
- Results visualization and storage

Author: Your Name
Date: 2024-02-20
"""

# Import required libraries
import os
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (RocCurveDisplay, accuracy_score, precision_score,
                           recall_score, f1_score, roc_auc_score, classification_report)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('churn_library')

# For matplotlib in non-GUI environments
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

def import_data(pth):
    """
    Import data from a CSV file.
    
    Args:
        pth (str): Path to the CSV file

    Returns:
        df: Pandas DataFrame containing the imported data
    """
    try:
        df = pd.read_csv(pth)
        # Create 'Churn' column - assuming 'Attrition_Flag' is in the data
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        return df
    except FileNotFoundError:
        logger.error(f"Could not find file at path: {pth}")
        raise
    except Exception as err:
        logger.error(f"Error occurred while importing data: {err}")
        raise

def perform_eda(df):
    """
    Perform EDA on the provided DataFrame and save figures to images folder.
    
    Args:
        df: pandas DataFrame containing the data
    """
    try:
        # Create images directory if it doesn't exist
        os.makedirs("./images/eda", exist_ok=True)
        
        # Churn Distribution
        plt.figure(figsize=(20,10))
        df['Churn'].hist()
        plt.title('Churn Distribution')
        plt.savefig("./images/eda/churn_distribution.png")
        plt.close()
        
        # Customer Age Distribution
        plt.figure(figsize=(20,10))
        df['Customer_Age'].hist()
        plt.title('Customer Age Distribution')
        plt.savefig("./images/eda/customer_age_distribution.png")
        plt.close()

        # Marital Status Distribution
        plt.figure(figsize=(20,10))
        df['Marital_Status'].value_counts('normalize').plot(kind='bar')
        plt.title('Marital Status Distribution')
        plt.savefig("./images/eda/marital_status_distribution.png")
        plt.close()

        # Total Transaction Distribution
        plt.figure(figsize=(20,10))
        sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
        plt.title('Total Transaction Distribution')
        plt.savefig("./images/eda/total_transaction_distribution.png")
        plt.close()

        # Correlation Heatmap (only for numeric columns)
        plt.figure(figsize=(20,10))
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        sns.heatmap(df[numeric_columns].corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.title('Correlation Heatmap')
        plt.savefig("./images/eda/correlation_heatmap.png")
        plt.close()
        
        logger.info("EDA performed successfully")
        
    except Exception as err:
        logger.error(f"Error occurred during EDA: {err}")
        raise

def encoder_helper(df, category_lst, response):
    """
    Helper function to encode categorical variables using mean response encoding.
    
    Args:
        df: pandas DataFrame
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that could be used for naming variables or index y column]

    Returns:
        df: pandas DataFrame with new encoded columns
    """
    try:
        # Create a copy of the dataframe
        df_encoded = df.copy()
        
        # Encode each categorical column
        for category in category_lst:
            # Calculate mean response for each category value
            category_groups = df.groupby(category)[response].mean()
            
            # Create new column name
            new_column = f"{category}_{response}"
            
            # Map the mean responses to the new column
            df_encoded[new_column] = df[category].map(category_groups)
            
            logger.info(f"Encoded {category} as {new_column}")
            
        return df_encoded
        
    except Exception as err:
        logger.error(f"Error occurred during encoding: {err}")
        raise

def perform_feature_engineering(df, response):
    """
    Perform feature engineering steps and split the data into training and testing sets.
    
    Args:
        df: pandas DataFrame
        response: string of response name [optional argument that could be used for naming variables or index y column]

    Returns:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    """
    try:
        # Keep only relevant columns for modeling
        keep_cols = [
            'Customer_Age',
            'Dependent_count',
            'Months_on_book',
            'Total_Relationship_Count',
            'Months_Inactive_12_mon',
            'Contacts_Count_12_mon',
            'Credit_Limit',
            'Total_Revolving_Bal',
            'Avg_Open_To_Buy',
            'Total_Amt_Chng_Q4_Q1',
            'Total_Trans_Amt',
            'Total_Trans_Ct',
            'Total_Ct_Chng_Q4_Q1',
            'Avg_Utilization_Ratio',
            'Gender_Churn',
            'Education_Level_Churn',
            'Marital_Status_Churn',
            'Income_Category_Churn',
            'Card_Category_Churn'
        ]

        y = df[response]
        X = df[keep_cols]

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        logger.info("Feature engineering completed successfully")
        
        return X_train, X_test, y_train, y_test

    except Exception as err:
        logger.error(f"Error occurred during feature engineering: {err}")
        raise

def train_models(X_train, X_test, y_train, y_test):
    """
    Train, store and evaluate models.
    
    Args:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    """
    try:
        # Create directory for models if it doesn't exist
        os.makedirs("./models", exist_ok=True)
        os.makedirs("./images/results", exist_ok=True)

        # Scale features for Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Random Forest (using unscaled data)
        rfc = RandomForestClassifier(random_state=42)
        
        # Logistic Regression with scaled data
        lrc = LogisticRegression(max_iter=3000)

        # Grid search parameters for Random Forest
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

        # Grid search for Random Forest (using unscaled data)
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(X_train, y_train)

        # Train Logistic Regression on scaled data
        lrc.fit(X_train_scaled, y_train)

        # Save best models and scaler
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')
        joblib.dump(scaler, './models/scaler.pkl')
        logger.info("Models and scaler saved successfully")

        # Make predictions (using appropriate scaled/unscaled data)
        y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
        y_train_preds_lr = lrc.predict(X_train_scaled)
        y_test_preds_lr = lrc.predict(X_test_scaled)

        # Plot ROC curves (using appropriate scaled/unscaled data)
        plt.figure(figsize=(15, 8))
        RocCurveDisplay.from_estimator(cv_rfc.best_estimator_, X_test, y_test, name='Random Forest')
        RocCurveDisplay.from_estimator(lrc, X_test_scaled, y_test, name='Logistic Regression')
        plt.title('ROC Curves')
        plt.savefig('./images/results/roc_curves.png')
        plt.close()

        # Feature importance plot for Random Forest
        feature_importance = cv_rfc.best_estimator_.feature_importances_
        indices = np.argsort(feature_importance)[::-1]
        plt.figure(figsize=(20,8))
        plt.title("Feature Importance")
        plt.bar(range(X_train.shape[1]), feature_importance[indices])
        plt.xticks(range(X_train.shape[1]), [X_train.columns[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('./images/results/rf_feature_importance.png')
        plt.close()

        # Classification reports
        plt.figure(figsize=(8,6))
        plt.text(0.01, 1.0, str('Logistic Regression Train\n' + 
                classification_report(y_train, y_train_preds_lr)), 
                {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.5, str('Logistic Regression Test\n' + 
                classification_report(y_test, y_test_preds_lr)), 
                {'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig('./images/results/logistic_report.png')
        plt.close()

        plt.figure(figsize=(8,6))
        plt.text(0.01, 1.0, str('Random Forest Train\n' + 
                classification_report(y_train, y_train_preds_rf)), 
                {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.5, str('Random Forest Test\n' + 
                classification_report(y_test, y_test_preds_rf)), 
                {'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig('./images/results/rf_report.png')
        plt.close()

        logger.info("Model training and evaluation completed successfully")

    except Exception as err:
        logger.error(f"Error occurred during model training: {err}")
        raise

if __name__ == "__main__":
    # Import data
    df = import_data("./data/bank_data.csv")
    
    # Perform EDA
    perform_eda(df)
    
    # Define categorical columns and response
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    response = 'Churn'
    
    # Encode categorical variables
    df = encoder_helper(df, cat_columns, response)
    
    # Perform feature engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, response)
    
    # Train and evaluate models
    train_models(X_train, X_test, y_train, y_test)
    
    logger.info("Churn prediction pipeline completed successfully")
