# Deep-Learning-Challenge


## **Overview**

The nonprofit organization, Alphabet Soup, wanted a tool that can help it select applicants for funding with the best chance of success in their ventures. The purpose of the analysis was to create a binary classifier that could predict whether applicants will be successful if funded by Alphabet Soup. 

## **Results**

Data Processing:

       * The target variable the  model was ***IS_SUCCESSFUL*** column (indicates if the money was used effectively or not)


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
            
