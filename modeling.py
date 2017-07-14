import pandas as pd
import numpy as np

import cleaning_data


def get_X_y(made_data):

    X_cols = ['day_of_week', 'status', 'hour', 'distance_to_next',
            'speed', 'linear_ref']
    y_col = ['time_to_next']
    
    X = made_data[X_cols].values
    y = made_data[y_col].values
    
    return X, y


def get_modellers(path):
    
#    made_data = cleaning_data.clean_data(path)
    made_data = pd.read_csv("data/cleaned_potong1.csv.tar.gz")
    
    X, y = get_X_y(made_data)

    from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
    labelencoder = LabelEncoder()
    X[:, 1] = labelencoder.fit_transform(X[:, 1])
    
    onehotencoder = OneHotEncoder(categorical_features=[0])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:, 1:]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    
    #sc_y = StandardScaler()
    #y_train = sc_y.fit_transform(y_train)
    
    #import keras
    from keras.models import Sequential
    from keras.layers import Dense
    
    regressor = Sequential()
    
    regressor.add(Dense(output_dim=30, init='normal', activation='relu', input_dim=11))
    regressor.add(Dense(output_dim=30, init='normal', activation='relu'))
    regressor.add(Dense(output_dim=30, init='normal', activation='relu'))
    regressor.add(Dense(output_dim=30, init='normal', activation='relu'))
#    regressor.add(Dense(output_dim=50, init='normal', activation='relu'))
#    regressor.add(Dense(output_dim=50, init='normal', activation='relu'))
#    regressor.add(Dense(output_dim=50, init='normal', activation='relu'))
    regressor.add(Dense(output_dim=1, init='normal'))
    
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    regressor.fit(X_train, y_train, batch_size=32, nb_epoch=300)
    
#    y_pred = regressor.predict(X_test)
#    score = regressor.evaluate(X_test, y_test)

    return [regressor, labelencoder, onehotencoder, sc_X]

path = "data/pothong_1.csv"
[regressor, labelencoder, onehotencoder, sc_X] = get_modellers(path)




#y_test = np.array(y_test)
#y_pred = np.array(y_pred)
#
#y_diff = y_test - y_pred
#y_check = np.append(y_diff, y_pred, axis=1)



#from sklearn.linear_model import Ridge
#regressor = Ridge()
#regressor.fit(X_train, y_train)
#
#y_pred = regressor.predict(X_test)
#score = regressor.score(X_test, y_test)


#from sklearn.externals import joblib
#joblib.dump([X, y], 'saved-model/Xy.pkl')


def save_model(modellers):
    from sklearn.externals import joblib
    
    [regressor, labelencoder, onehotencoder, sc] = modellers
    
    # serialize model to JSON
    model_json = regressor.to_json()
    with open("saved-model/model.json", "w") as json_file:
        json_file.write(model_json)
        
    # serialize weights to HDF5
    regressor.save_weights("saved-model/model.h5")
    print("Saved model to disk")
    
    joblib.dump([labelencoder, onehotencoder, sc], 'saved-model/encoders.pkl')



#save_model([regressor, labelencoder, onehotencoder, sc_X])


#from keras.models import model_from_json
#
## load json and create model
#json_file = open('saved-model/model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
## load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")
#
#
#
#yy_pred = loaded_model.predict(X_test)



