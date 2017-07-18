# ## Import Libraries & Data
import pandas as pd
import numpy as np

def clean_data(path):
    
    def clean_status_and_trip_id(data):
        """ Clean incorrect inbound/outbound and add trip_id for the groups of 
            contiguous points with the same status """ 
        
        def get_linear_ref(dt):
            return dt['linear_outbound'] if (dt['status'] == 'outbound') else dt['linear_inbound']
        
        data['linear_ref'] = data.apply(get_linear_ref, axis=1)
#        def is_next_trip(prev, curr):
#            if prev[']
        
        status = []
        trip_id = []
        count_trip = 0
        last_status = ""
#        last_linear_ref = 0
        for i, point in data.iterrows():
            lower = i-5 if i-5>=0 else 0
            upper = i+6 if i+6<=data.shape[0] else data.shape[0]
#            print(data.iloc[lower:upper, :]['status'])
            status += [data.iloc[lower:upper, :]['status'].value_counts().idxmax()]
            
#            linref_diff = dt['linear_outbound'] if (dt['status'] == 'outbound') else dt['linear_inbound']
            
            prev_index = i-1 if i-1 >= 0 else 0
            if((status[-1] != last_status)  | 
                    (point['linear_ref'] - data.loc[prev_index, 'linear_ref'] > 0.6) |
                    ((point['gps_timestamp'] - data.loc[prev_index, 'gps_timestamp']).seconds > 500) ):
                last_status = point['status']
                count_trip += 1
            trip_id += [count_trip]
            
#            if(point['status'] == "inbound"):
#                if(point['linear_inbound'] - data.loc[i-1])
        
        data['status'] = status
        data['trip_id'] = trip_id
        
        return data    

    def include_next_point(data, n):
        """ Create new dataframe for 2 GPS points combined (current point 
        and next point, which provides time to next point to predict) """
        def get_distance(df):
            if df['status'] == 'inbound':
                return df['next_lin_in'] - df['linear_inbound']
            if df['status'] == 'outbound':
                return df['next_lin_out'] - df['linear_outbound']
        
        def get_linear_ref(dt):
            return dt['linear_inbound'] if dt['status'] == 'inbound' else dt['linear_outbound']
            
        temp_data = data.copy()
        temp_data['next_time'] = temp_data['gps_timestamp'].shift(-n)
        temp_data['next_lin_in'] = temp_data['linear_inbound'].shift(-n)
        temp_data['next_lin_out'] = temp_data['linear_outbound'].shift(-n)
        temp_data['next_status'] = temp_data['status'].shift(-n)
        temp_data['next_trip_id'] = temp_data['trip_id'].shift(-n)
        
        temp_data['next_index'] = temp_data['index'].shift(-n)
        
        temp_data = temp_data[temp_data['trip_id']==temp_data['next_trip_id']]
        
        temp_data['time_to_next'] = temp_data['next_time'] - temp_data['gps_timestamp']
        temp_data = temp_data[temp_data['status'] == temp_data['next_status']]
        temp_data['distance_to_next'] = temp_data.apply(get_distance, axis=1)
        
        
        
        temp_data['hour'] = temp_data['gps_timestamp'].apply(lambda dt: dt.hour)
        temp_data['day_of_week'] = temp_data['gps_timestamp'].apply(lambda dt: dt.weekday())
        temp_data['linear_ref'] = temp_data.apply(get_linear_ref, axis=1)
        
        
        return temp_data
    
    
    raw_data = pd.read_csv(path)
    print("data loaded")
    data = raw_data.copy()
    
    
    data = data[["gps_timestamp", "speed", "vehicle_id", "linear_inbound", 
             "linear_outbound", "longitude", "latitude", "bus_line", "status"]]
    
    data = data[(data['status'] == 'inbound') | (data['status'] == 'outbound')]
    
    data = data.iloc[::150, :]
    
    index = data.index.values
    data = data.reset_index(drop=True)
    data['index'] = index

    data['gps_timestamp'] = pd.to_datetime(data['gps_timestamp'])

    print("Start cleaning status and adding trip id")
    data = clean_status_and_trip_id(data)
    print("status cleaned and trip id added")
    
    
    # Create data for training by combining 2 point
    print("combining data...")
    made_data = pd.DataFrame()
    for t in range(3, 100):
        print("combining every", t, "point")
        temp = include_next_point(data, t)
        made_data = pd.concat([made_data, temp])
    
    made_data = made_data[made_data['distance_to_next'] >= 0]
    made_data['time_to_next'] = made_data['time_to_next'].dt.seconds

    return made_data, data


cleaned_data, data = clean_data("data/pothong_2.csv.gz")
print("data cleaned!")
cleaned_data.to_csv("data/cleaned_potong2.csv.gz", compression='gzip', index=False)
print("saved")


#raw_data = pd.read_csv("data/pothong_1.csv.gz")
#print("data loaded")
#data = raw_data.copy()
#
#
#data = data[["gps_timestamp", "speed", "vehicle_id", "linear_inbound", 
#         "linear_outbound", "longitude", "latitude", "bus_line", "status"]]
#
#data = data[(data['status'] == 'inbound') | (data['status'] == 'outbound')]
#
#data = data.iloc[::150, :]

