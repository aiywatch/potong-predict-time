from sklearn.externals import joblib
import pandas as pd
import datetime
import math
import requests
from keras.models import model_from_json
from geopy.distance import vincenty


BUS_LINES = ['1', '2', '2a', '3']

ROUTE = pd.DataFrame({'1': [586, 585],
                      '2': [584, 583],
                      '2a': [587, 588],
                      '3': [582, 581]}, index=['in', 'out'])


def import_model(bus_line):
    MODEL_PATH = "saved-model"
    [labelencoder, onehotencoder, sc] = joblib.load("{}/{}/encoders.pkl".format(MODEL_PATH, bus_line))
    
    # load json and create model
    json_file = open("{}/{}/model.json".format(MODEL_PATH, bus_line), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    regressor = model_from_json(loaded_model_json)
    # load weights into new model
    regressor.load_weights("{}/{}/model.h5".format(MODEL_PATH, bus_line))
    print("Loaded model from disk")

    return [regressor, labelencoder, onehotencoder, sc]


def get_lastest_gps(bus_vehicle_id):
    data = requests.get('https://api.traffy.xyz/vehicle/?vehicle_id='+str(bus_vehicle_id)).json()
    bus = data['results']
    if(bus):
        return bus[0]
    return None

def get_bus_info(bus):
    bus_data = pd.Series()
    bus_data['vehicle_id'] = bus['vehicle_id']
    bus_data['timestamp'] = bus['info']['gps_timestamp']
    bus_data['name'] = bus['name']
#    bus_data['trip_id'] 
    bus_data['linear_ref'] = bus['checkin_data']['route_linear_ref']
    bus_data['speed'] = bus['info']['speed']
    bus_data['direction'] = bus['info']['direction']
    bus_data['status'] = bus['status']
    [bus_data['lon'], bus_data['lat']] = bus['info']['coords']
#    bus_data['route_length_in_meter']
#    bus_data['distance_from_route_in_meter']
    return bus_data

def clean_data(data_point, usr_linear_ref):
    new_data_point = data_point.copy()
    new_data_point['timestamp'] = pd.to_datetime(new_data_point['timestamp'])
    new_data_point['day_of_week'] = new_data_point['timestamp'].weekday()
    new_data_point['hour'] = new_data_point['timestamp'].hour
    new_data_point['direction'] = 'outbound' if new_data_point['direction'] == 'out' else 'inbound'
   
    
    new_data_point['distance_to_next'] = usr_linear_ref - new_data_point['linear_ref']
    

    new_data_point = new_data_point[['day_of_week', 'direction', 'hour', 'distance_to_next',
                                     'speed', 'linear_ref']]
    return new_data_point

def encode_data(data_point, labelencoder, onehotencoder, sc):
    new_data_point = data_point.copy()
    
    new_data_point[1] = labelencoder.transform([new_data_point[1]])[0]
#    new_data_point[0] = labelencoder.transform([new_data_point[0]])[0]
    
    new_data_point = onehotencoder.transform([new_data_point]).toarray()
    new_data_point = new_data_point[0, 1:]
    
    new_data_point = sc.transform([new_data_point])
    return new_data_point

#def predict_time(bus_line, bus_vehicle_id, usr_lat, usr_lon, usr_dir):
##    print('predicting')
#    bus_line = str(bus_line)
#    if(bus_line not in BUS_LINES):
#        return 'This bus line is not available!'
#        
##    print(pd.to_datetime(datetime.datetime.utcnow()))
#    bus = get_lastest_gps(bus_vehicle_id)
#
##    print(bus)
#    if(not bus):
#        return 'This bus is not available!'
#    
#    [regressor, labelencoder, onehotencoder, sc] = import_model(bus_line)
#    
#    bus_data = get_bus_info(bus)
#    
#    
#    
#    
#    route_id = ROUTE.loc[usr_dir, bus_line]
#    data = requests.get('https://api.traffy.xyz/v0/route/{}/linear_ref/?coords={},{}'.format(
#            route_id, usr_lon, usr_lat)).json()
#    usr_linear_ref = data['location']['linear_ref']
#    
#    
#    time_now = pd.to_datetime(datetime.datetime.utcnow())
#    
#    cleaned_bus_data = clean_data(bus_data, time_now, usr_linear_ref)
##    print(cleaned_bus_data)
#    
#    encoded_bus_data = encode_data(cleaned_bus_data, labelencoder, onehotencoder, sc)
#    
#    predicted_time = regressor.predict([encoded_bus_data])
##    print('predicted_time', predicted_time)
#    
#    
#    output = {'predicted_time': predicted_time[0],
#              'predicting time': time_now,
#                    'last_point_data': {
#                        'user_linear_ref': usr_linear_ref,
#                        'last_timestamp': bus_data['timestamp'],
#                        'timestamp_now': time_now,
#                        'distance_to_next': cleaned_bus_data['distance_to_next'],
#                        'last_linear_ref': bus_data['linear_ref'],
#                        'last_speed': bus_data['speed'],
#                        'direction': bus_data['direction'],}
#                    }
##    print(output)
#    return output


def predict_time(bus_data, usr_linear_ref):
    
    time_now = pd.to_datetime(datetime.datetime.utcnow())
    
    [regressor, labelencoder, onehotencoder, sc] = import_model(bus_line)
    cleaned_bus_data = clean_data(bus_data, usr_linear_ref)
    encoded_bus_data = encode_data(cleaned_bus_data, labelencoder, onehotencoder, sc)
    
    predicted_time = regressor.predict([encoded_bus_data])
    
    output = {'predicted_arrival_time': float(predicted_time[0][0]),
                    'last_point_data': {
                        'user_linear_ref': usr_linear_ref,
                        'last_timestamp': bus_data['timestamp'],
                        'timestamp_now': time_now,
                        'distance_to_next': cleaned_bus_data['distance_to_next'],
                        'last_linear_ref': bus_data['linear_ref'],
                        'last_speed': bus_data['speed'],
                        'direction': bus_data['direction'],
                        'bus_name': bus_data['name'],}
                    }
    return output


def request_prediction(bus_line, usr_lat, usr_lon, usr_dir):
    
    data = requests.get('https://api.traffy.xyz/vehicle/?line=potong-{}'.format(
            bus_line)).json()
    buses = data['results']
    
    if not buses:
        return 'This bus line is not available!'
    
    bus_df = pd.DataFrame(columns=['vehicle_id', 'timestamp', 'name', 'linear_ref', 
                              'speed', 'direction', 'lat', 'lon', 'status'])
    
    for bus in buses:
        bus_info = get_bus_info(bus)
        bus_df = bus_df.append(bus_info, ignore_index=True)
    

    route_id = ROUTE.loc[usr_dir, bus_line]
    data = requests.get('https://api.traffy.xyz/v0/route/{}/linear_ref/?coords={},{}'.format(
            route_id, usr_lon, usr_lat)).json()
    usr_linear_ref = data['location']['linear_ref']
    
#    print('usr_linref', usr_linear_ref)
    
    running_buses = bus_df[(bus_df['status'] == usr_dir) & 
                           (bus_df['linear_ref'] < usr_linear_ref)
                          ].sort_values('linear_ref', ascending=False)
    
#    print(running_buses)
    if running_buses.shape[0] == 0:
        return 'No bus is available yet'
    
    closest_bus = running_buses.iloc[0, :]
    
    return predict_time(closest_bus, usr_linear_ref)


bus_line = '1'
#usr_lon = 98.3663
#usr_lat = 7.89635
usr_lon = 98.401550
usr_lat = 7.862933

usr_dir = 'out'
request_prediction(bus_line, usr_lat, usr_lon, usr_dir)



#bus_vehicle_id_1 = 359739072722465
#bus_vehicle_id_2 = 359739072730088
#bus_vehicle_id_2a = 352648061891537
#bus_vehicle_id_3 = 358901049778803
#
#
##usr_lat = 98.3663
##usr_lon = 7.89635
#
#usr_lat = 98.3633795
#usr_lon = 7.858667
#
##usr_lat = 98.395044
##usr_lon = 7.8677455
##
##usr_lat = 98.36688516666666
##usr_lon = 7.887917666666667
#
#cleaned_bus_data = predict_time(2, bus_vehicle_id_2, usr_lat, usr_lon, 'in')














