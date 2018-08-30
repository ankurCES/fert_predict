import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import model_from_json

def rnn_lstm(layers, params):

	model = Sequential()
	model.add(LSTM(input_shape=(layers[1], layers[0]), return_sequences=True, units=layers[1]))
	model.add(Dropout(params['dropout_keep_prob']))
	model.add(LSTM(layers[2], return_sequences=False))
	model.add(Dropout(params['dropout_keep_prob']))
	model.add(Dense(units=layers[3]))
	model.add(Activation("tanh"))

	model.compile(loss="mean_squared_error", optimizer="rmsprop")
	return model

def predict_next_timestamp(model, history):

	prediction = model.predict(history)
	prediction = np.reshape(prediction, (prediction.size,))
	return prediction 

def load_model(model_file_path, model_weight_path):
	# load json and create model
	json_file = open(model_file_path, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(model_weight_path)
	loaded_model.compile(loss="mean_squared_error", optimizer="rmsprop")
	print("Loaded model from disk")
	return loaded_model