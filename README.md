# Alphabet Soup Charity Success Prediction

## Overview

Alphabet Soup, a nonprofit foundation, wants a tool to identify applications with the highest chance of success in their ventures. Using historical funding data for over 34,000 organization, this project develops a binary classification model to predict whether an applicant will be successful if funded.

The dataset includes the following key features:
* **EIN and NAME**—Identification columns
* **APPLICATION_TYPE**—Alphabet Soup application type
* **AFFILIATION**—Affiliated sector of industry
* **CLASSIFICATION**—Government organization classification
* **USE_CASE**—Use case for funding
* **ORGANIZATION**—Organization type
* **STATUS**—Active status
* **INCOME_AMT**—Income classification
* **SPECIAL_CONSIDERATIONS**—Special considerations for application
* **ASK_AMT**—Funding amount requested
* **IS_SUCCESSFUL**—Was the money used effectively

## Project Workflow
**Step 1: Preprocess the Data**
* Target: `IS_SUCCESSFUL`
* Features: `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`
* Removed irrelevant columns: `EIN`, `NAME`
* Combined rare categorical values into `OTHER`
* Encoded categorical variables using `pd.get_dummies()`
* Split the data into training and testing sets using `train_test_split`
* Scaled features using `StandardScaler()`

**Step 2: Compile, Train, and Evaluate the Model**
* Designed a neural network with TensorFlow/Keras
* Optimized hyperparameters:
  * **Activation function:** `tanh`
  * **Number of hidden layers:** 4
  * **Units in hidden layers:** 7 → 27 → 9 → 5
  * **Number of epochs:** 20
  * **Initial epoch:** 7
* Added a callback to save model weights every 5 epochs
* Evaluated the model on the testing data (loss and accuracy)

**Step 3: Optimize the Model**
* Attempted to improve performance using hyperparameter tuning:
  * Created a function to automatically iterate through possible hidden layers, units, and activation functions
  * Generated the best hyperparameters for the model
* **Reuslts:** Model accuracy slightly increased by 0.0028, achieving ~73% accuracy on the test set
* The target accuracy of 75% was not achieved

## Summary
* The neural network is correctly predicting funding outcomes approximately73% of the time
* The model effectively identifies successful applications but did not reach the 75% target accuracy
* **Recommendation:** Consider alternative classification models (e.g., Random Forest) or improve feautres to boost predictive performance


