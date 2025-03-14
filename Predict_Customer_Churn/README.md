# Predict Customer Churn with Clean Code

## Project Overview
In this project, I will identify credit card customers who are most likely to churn. The goal is to implement best coding and engineering practices while developing a machine learning package that:
- Follows **PEP8 coding standards**
- Is **modular, documented, and tested**
- Can be run **interactively or from the command-line interface (CLI)**

This project helps me practice **testing, logging, and code refactoring**, which are crucial for maintaining production-level machine learning applications.

## File Structure
```
.
├── Guide.ipynb                          # Getting started and troubleshooting tips
├── churn_notebook.ipynb                 # Provided: Code to be refactored
├── churn_library.py                      # ToDo: Define the functions
├── churn_script_logging_and_tests.py    # ToDo: Implement tests and logging
├── README.md                             # ToDo: Project documentation
├── data                                  # Folder containing the dataset
│   └── bank_data.csv                     # Dataset for churn prediction
├── images                                # Folder storing EDA results
│   ├── eda
│   └── results
├── logs                                  # Folder for logs
└── models                                # Folder storing trained models
```

## Running the Project
### 1. Install Dependencies
I need to make sure all required dependencies are installed. I can do this using:
```bash
pip install -r requirements.txt
```

### 2. Run Churn Library
To execute the churn library and generate predictions, I run:
```bash
ipython churn_library.py
```

### 3. Run Tests and Logging
I can ensure that testing and logging are functional using:
```bash
ipython churn_script_logging_and_tests.py
```

This will generate logs in the `/logs` folder and validate the correctness of functions.

## Code Quality Standards
To maintain clean and efficient code, I will follow these best practices:
- **Style Guide:** Format code using **PEP8**.
```bash
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_library.py
```
- **Static Code Analysis:** Check for errors and improvements using **Pylint**.
```bash
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```
I should aim for a **Pylint score above 7**.

## Project Expectations
By the end of the project, I should have completed these files:
1. `churn_library.py` – Implements functions for churn prediction
2. `churn_script_logging_and_tests.py` – Logs errors and contains test cases
3. `README.md` – Provides project overview and usage instructions

I will make sure all **functions and files have proper docstrings**, explaining:
- **Purpose of the function/file**
- **Input and output parameters**
- **Author and date of creation**

Happy coding! 🚀

