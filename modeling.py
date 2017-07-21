import pandas as pd
import numpy as np

#import cleaning_data

SAVED_MODEL_PATH = 'saved-model/'


def get_X_y(made_data):

    X_cols = ['day_of_week', 'status', 'hour', 'distance_to_next',
            'speed', 'linear_ref']
    y_col = ['time_to_next']
    
    X = made_data[X_cols].values
    y = made_data[y_col].values
    
    return X, y


def get_modellers(bus_line):
    
    DATA_PATH = "data/cleaned_potong{}.csv.gz".format(bus_line)
#    made_data = cleaning_data.clean_data(path)
    made_data = pd.read_csv(DATA_PATH)
    made_data = made_data.dropna()
    
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
    
    #import keras
    from keras.models import Sequential
    from keras.layers import Dense
    
    feature_num = X.shape[1]
    
    regressor = Sequential()
    
    regressor.add(Dense(output_dim=30, init='normal', activation='relu', input_dim=feature_num))
    regressor.add(Dense(output_dim=30, init='normal', activation='relu'))
    regressor.add(Dense(output_dim=30, init='normal', activation='relu'))
    regressor.add(Dense(output_dim=1, init='normal'))
    
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    regressor.fit(X_train, y_train, batch_size=32, nb_epoch=150)
    
    
#    y_pred = regressor.predict(X_test)
#    score = regressor.evaluate(X_test, y_test)

    return [regressor, labelencoder, onehotencoder, sc_X, X_test, y_test]




def save_model(modellers, bus_line):
    from sklearn.externals import joblib
    
    [regressor, labelencoder, onehotencoder, sc] = modellers
    
    # serialize model to JSON
    model_json = regressor.to_json()
    with open("{}{}/model.json".format(SAVED_MODEL_PATH, bus_line), "w") as json_file:
        json_file.write(model_json)
        
    # serialize weights to HDF5
    regressor.save_weights("{}{}/model.h5".format(SAVED_MODEL_PATH, bus_line))
    print("Saved model to disk")
    
    joblib.dump([labelencoder, onehotencoder, sc], "{}{}/encoders.pkl".format(SAVED_MODEL_PATH, bus_line))
    joblib.dump([X_test, y_test], "{}{}/Xy.pkl".format(SAVED_MODEL_PATH, bus_line))


#BUS_LINE = '3'
#
#[regressor, labelencoder, onehotencoder, sc_X, X_test, y_test] = get_modellers(BUS_LINE)
#save_model([regressor, labelencoder, onehotencoder, sc_X], BUS_LINE)


def run(bus_line):
    [regressor, labelencoder, onehotencoder, sc_X, X_test, y_test] = get_modellers(bus_line)
    save_model([regressor, labelencoder, onehotencoder, sc_X], bus_line)



run('1')
run('2')
run('2a')
run('3')



