import sys, getopt
import json
import model_helper
import data_helper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def train(data, model_file, model_weights_file):
	parameter_file = 'training_config.json'

	params = json.loads(open(parameter_file).read())

	x_train, y_train, x_test, y_test, x_test_raw, y_test_raw, last_window_raw, last_window = data_helper.load_timeseries(data, params)

	model_file_path = Path(model_file)
	if model_file_path.is_file():
		print('Existing trained model found.')
		model = model_helper.load_model(model_file, model_weights_file)
	else:
		print('Creating a new model.')
		lstm_layer = [1, params['window_size'], params['hidden_unit'], 1]
		model = model_helper.rnn_lstm(lstm_layer, params)

	model.fit(
		x_train,
		y_train,
		batch_size=params['batch_size'],
		epochs=params['epochs'],
		validation_split=params['validation_split'],
		verbose=1)

	predicted = model_helper.predict_next_timestamp(model, x_test)        
	predicted_raw = []
	for i in range(len(x_test_raw)):
		predicted_raw.append((predicted[i] + 1) * x_test_raw[i][0])
	
	# serialize model to JSON
	model_json = model.to_json()
	with open(model_file, 'w') as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(model_weights_file)
	print("Saved model to disk")

def predict(data, model, number_of_predictions):
	parameter_file = 'training_config.json'

	params = json.loads(open(parameter_file).read())
	predicted_values=[]
	for i in range(number_of_predictions):
		x_train, y_train, x_test, y_test, x_test_raw, y_test_raw, last_window_raw, last_window = data_helper.load_timeseries(data, params)
		next_timestamp = model_helper.predict_next_timestamp(model, last_window)
		next_timestamp_raw = (next_timestamp[0] + 1) * last_window_raw[0][0]
		print('#{}#The next time stamp forecasting is: {}'.format(i+1, next_timestamp_raw))
		predicted_values.append(next_timestamp_raw)
		np_pred = np.array([next_timestamp_raw])
		data = np.append(data, np_pred)

def main(argv):
	try:
		opts, args = getopt.getopt(argv, 'hm:d', ['help', 'file=', 'dataset=', 'mode='])
	except getopt.GetoptError:
		print('\nUSAGE:')
		console.success('process_data.py --dataset <dataset_name>')
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-h', '--help'):
			print('\nUSAGE:')
			print('process_data.py --dataset <dataset_name> --file <file_name> --mode <mode>')
			sys.exit()
		elif opt in ('-f', '--file'):
			file_name = arg
		elif opt in ('-m', '--mode'):
			if arg in ['train', 'predict']:
				mode = arg
			else:
				mode = None
		elif opt in ('-d', '--dataset'):
			dataset_name = arg
		else:
			print('Invalid options. Run with --help to see usage')
	
	series = pd.read_csv(file_name, sep=',', header=0, index_col=0, squeeze=True)
	data = series.values

	model_file = 'models/'+dataset_name+'.json'
	model_weights_file = 'models/'+dataset_name+'.h5'

	if mode == 'train':
		train(data, model_file, model_weights_file)
	elif mode == 'predict':
		model = model_helper.load_model(model_file, model_weights_file)
		predict(data, model, 3)
	else:
		print('Invalid options. Run with --help to see usage')


if __name__ == '__main__':
	main(sys.argv[1:])