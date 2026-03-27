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
      
* Unforunately the model was not able to reach the target performance of 75% 

* To increase model perforamnce, I created a function that automcatically runs through each possible hidden layer, units, and activitation combination possible. It then generates the best hyperparamters for the model (see borrowed code section).


## **Summary**

After optimizing the model, the overall accuracy score continued to be in the same range with a slight increase of .0028. This indicates that the neural network model is correctly predicting the outcomes approximately 73% of the time on the test dataset

## **Borrowed Code**

The code below was borrowed from a class session showing how to create and find the best hyperparameters for the nerual networks model automatically.
        
        def create_model(hp): # Function used to run through every possible hidden layer and activiation combination possible
            nn = tf.keras.models.Sequential()
            number_of_features = len(X_train.columns)  # Number of columns in the dataset; This works if X_train is a DataFrame
            hidden_layer1 = 80
            hidden_layer2 = 30
            activation = hp.Choice('activation',['relu','tanh','sigmoid'])
            
            nn.add(tf.keras.layers.Dense(units=hp.Int('first_units',
                                              min_value=1,
                                              max_value=80,
                                              step=2),
                                 activation=activation,
                                 input_dim=number_of_features)) # Created to find most optimial first hidden layer

            for i in range(hp.Int('num_layers', 1, 6)):
                nn.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                                                          min_value=1,
                                                          max_value=30,
                                                          step=2),
                                             activation=activation)) #Create to find most optimial second hidden layer

            nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

            nn.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

            return nn
            
