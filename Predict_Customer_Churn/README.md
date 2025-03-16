# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project implements a machine learning model to identify credit card customers that are most likely to churn. The project includes a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested).

The project contains a library of functions `churn_library.py` that implements the ML pipeline, and a testing file `churn_script_logging_and_tests.py` that tests the functions and logs any errors.

## Files and Data Description
```
.
├── Guide.ipynb                     # Getting started guide notebook
├── churn_notebook.ipynb           # Contains the original notebook
├── churn_library.py              # Main library of functions
├── churn_script_logging_and_tests.py    # Testing and logging functions
├── README.md                     # Project documentation
├── requirements.txt              # Required packages
├── data/                        # Data directory
│   └── bank_data.csv           # Dataset used for the project
├── images/                      # Store all generated images
│   ├── eda/                    # Exploratory Data Analysis plots
│   └── results/                # Model results plots
├── logs/                       # Store all logs
│   └── churn_library.log      # Logging results
└── models/                     # Store model files
    ├── logistic_model.pkl     # Logistic Regression model
    └── rfc_model.pkl          # Random Forest model
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/predict-customer-churn.git
cd predict-customer-churn
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Running Files

The project contains two main files that can be executed:

### 1. churn_library.py
This is the main library containing all functions needed for the churn prediction model. While it's primarily meant to be imported as a module, you can run it directly:

```bash
python churn_library.py
```

This will:
- Load and analyze the data
- Perform EDA and save visualizations
- Engineer features
- Train and save models
- Generate model performance plots

### 2. churn_script_logging_and_tests.py
This file contains unit tests for all functions in churn_library.py. Run it to verify everything works correctly:

```bash
python churn_script_logging_and_tests.py
```

This will:
- Test all functions in churn_library.py
- Log success/failure for each test in ./logs/churn_library.log
- Generate necessary images and model files

## Expected Outputs

After running the files, you should see:

1. In `./images/eda/`:
   - Churn distribution plot
   - Correlation heatmap
   - Other EDA visualizations

2. In `./images/results/`:
   - ROC curves plot
   - Feature importance plot
   - Classification reports for both models

3. In `./models/`:
   - `logistic_model.pkl`
   - `rfc_model.pkl`

4. In `./logs/`:
   - `churn_library.log` containing all test results and execution logs

## Testing

The testing framework includes tests for:
- Data import
- EDA functionality
- Feature engineering
- Model training and evaluation

To run tests and check the logs:
```bash
python churn_script_logging_and_tests.py
cat ./logs/churn_library.log  # View test results
```

## Author
Kashad Turner-Warren

## License
This project is licensed under the MIT License - see the LICENSE file for details
