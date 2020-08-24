from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

class NN:
    def __init__(self, model, scaler_x, scaler_y, important_bands, band_of_interest, prev_time_measurements, pos_measurements_x, pos_measurements_y):
        """
        Parameters
        ----------
        model : keras model
            trained (or untrained) keras model
        scaler_x : sklearn.preprocessing.MinMaxScaler
            scaling of the data beforehand
        scaler_y : sklearn.preprocessing.MinMaxScaler
            scaling of the data after predictiong
        important_bands: list
            bands used to predict the result. Usually a list of just
            one element which is the same as band_of_interest
        band_of_interest : int 
            band we are trying to predict
        prev_time_measurements : int
            the number of previous time measurements used to predict
        pos_measurements_x : int
            the number of x measurements
        pos_measurements_y : int
            the number of y measurements
        """
        
        self.model = model
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.important_bands = important_bands
        self.band_of_interest = band_of_interest
        self.prev_time_measurements = prev_time_measurements
        self.pos_measurements_x = pos_measurements_x
        self.pos_measurements_y = pos_measurements_y
        self.pos_measurements = self.pos_measurements_x * self.pos_measurements_y

def nn7():
    """ 
    Our pretrained network for predicting band 7 goes-16 data. 
    The network is a dense 50, 25, 1 feed forward network with relu 
    then linear activation.
    """

    dirname = os.path.dirname(__file__)
    model = keras.models.load_model(os.path.join(dirname, "band_7_predictor", "band_7_predictor"))
    with open(os.path.join(dirname, "band_7_predictor", "scaler_x.pkl"), "rb") as f:
        scaler_x = pickle.load(f)
    with open(os.path.join(dirname, "band_7_predictor", "scaler_y.pkl"), "rb") as f:
        scaler_y = pickle.load(f)
    nn7 = NN(model, scaler_x, scaler_y, [7], 7, 4, 5, 5)

    return nn7

def nn14():
    """ Our pretrained network for predicting band 14 goes-16 data. 
    The network is a dense 50, 25, 1 feed forward network with relu 
    then linear activation.  """

    dirname = os.path.dirname(__file__)
    model = keras.models.load_model(os.path.join(dirname, "band_14_predictor", "band_14_predictor"))
    with open(os.path.join(dirname, "band_14_predictor", "scaler_x.pkl"), "rb") as f:
        scaler_x = pickle.load(f)
    with open(os.path.join(dirname, "band_14_predictor", "scaler_y.pkl"), "rb") as f:
        scaler_y = pickle.load(f)
    nn14 = NN(model, scaler_x, scaler_y, [14], 14, 4, 5, 5)

    return nn14