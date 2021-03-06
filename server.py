
#from predict_potong_flask import predict_location
import predict_potong_time_flask.request_prediction as request_prediction
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict_potong_arrival_time/<bus_line>/<usr_lat>/<usr_lon>/<usr_dir>', methods=['GET'])
def predict_arrival_time(bus_line, usr_lat, usr_lon, usr_dir):
    return jsonify(request_prediction(bus_line, usr_lat, usr_lon, usr_dir))


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 8001


    app.run(host='0.0.0.0', port=port, debug=True)