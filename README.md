# Deep-Learning-Challenge


## **Overview**

The nonprofit organization, Alphabet Soup, wanted a tool that can help it select applicants for funding with the best chance of success in their ventures. The purpose of the analysis was to create a binary classifier that could predict whether applicants will be successful if funded by Alphabet Soup. 

## **Results**

*Data Processing:*

* The target variable the  model was ***IS_SUCCESSFUL*** column (indicates if the money was used effectively or not)
* Features used to compile the model:
   * Application type
   * Affiliation - Sector of industry
   * Classification - Government organization classification
   * Use Case - Reason for funding
   * Organization - Organization Type
   * Status
   * Income Amount
   * Special Considerations
   * Ask Amount
 * Variables that were removed from input data were the identification columns (EIN and Name columns) because they are neither possible features or target variables

*Compiling, Training, and Evaluating the Model*

How many neurons, layers, and activation functions did you select for your neural network model, and why?
* The model was optimized with the following hyperparameters:
    * Activation: tanh
    * Number of hidden layers: 4
    * First layer units: 37
    * Units in hidden layers(in order): 7, 27, 9, 5
    * Number of epochs: 20
    * Initial epoch: 7
*
* It produced a slightly higher accuracy rating than the previous model. When running the auto tuner, it gave those numbers and activation with the highest accuracy
Were you able to achieve the target model performance?
	No
What steps did you take in your attempts to increase model performance?
Ran an auto tuner…explain what auto tuner is



## **Summary**


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
            
