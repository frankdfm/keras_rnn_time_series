from __future__ import print_function
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN

from helper import *

# Create model
def create_fc_model(length):
	##### YOUR MODEL GOES HERE #####
	model = model_fc_factory(length)
	return model

def create_rnn_model(length, stateful, batch_size):
	##### YOUR MODEL GOES HERE #####
	model = model_simplernn_factory(length, stateful, batch_size)
	return model

def create_lstm_model(length, stateful, batch_size):
	##### YOUR MODEL GOES HERE #####
	return model_lstm_factory(length, stateful, batch_size)

# split train/test data
def split_data(x, y, ratio=0.8):
	to_train = int(len(x.index) * ratio)
	# tweak to match with batch_size
	to_train -= to_train % batch_size

	x_train = x[:to_train]
	y_train = y[:to_train]
	x_test = x[to_train:]
	y_test = y[to_train:]

	# tweak to match with batch_size
	to_drop = x.shape[0] % batch_size
	if to_drop > 0:
		x_test = x_test[:-1 * to_drop]
		y_test = y_test[:-1 * to_drop]

	# some reshaping
	##### RESHAPE YOUR DATA BASED ON YOUR MODEL #####

	return (x_train, y_train), (x_test, y_test)

# predicting parameters passed to "model.predict(...)"
batch_size = 1

# The input sequence min and max length that the model is trained on for each output point
min_length = 1
max_length = 10

# load data from files
noisy_data = np.loadtxt('../filter_data/noisy_data.txt', delimiter='\t', dtype=np.float)
smooth_data = np.loadtxt('../filter_data/smooth_data.txt', delimiter='\t', dtype=np.float)

print('noisy_data shape:{}'.format(noisy_data.shape))
print('smooth_data shape:{}'.format(smooth_data.shape))
print('noisy_data first 5 data points:{}'.format(noisy_data[:5]))
print('smooth_data first 5 data points:{}'.format(smooth_data[:5]))


# List to keep track of root mean square error for different length input sequences
fc_rmse_list=list()
rnn_stateful_rmse_list=list()
rnn_stateless_rmse_list=list()
lstm_stateful_rmse_list=list()
lstm_stateless_rmse_list=list()

for num_input in range(min_length,max_length+1):
	length = num_input

	print("*" * 33)
	print("INPUT DIMENSION:{}".format(length))
	print("*" * 33)

	# convert numpy arrays to pandas dataframe
	data_input = pd.DataFrame(noisy_data)
	expected_output = pd.DataFrame(smooth_data)

	# when length > 1, arrange input sequences
	if length > 1:
		##### ARRANGE YOUR DATA SEQUENCES #####
		data_input, expected_output = prepare_data_for_ts_training(data_input, expected_output, length)


	print('data_input length:{}'.format(len(data_input.index)) )

	# Split training and test data: use first 80% of data points as training and remaining as test
	(x_train, y_train), (x_test, y_test) = split_data(data_input, expected_output)
	print('x_train.shape: ', x_train.shape)
	print('y_train.shape: ', y_train.shape)
	print('x_test.shape: ', x_test.shape)
	print('y_test.shape: ', y_test.shape)

	print('Input shape:', data_input.shape)
	print('Output shape:', expected_output.shape)
	print('Input head: ')
	print(data_input.head())
	print('Output head: ')
	print(expected_output.head())
	print('Input tail: ')
	print(data_input.tail())
	print('Output tail: ')
	print(expected_output.tail())
	
	# Create the models and load trained weights
	print('Creating Fully-Connected Model and Loading Trained Weights...')
	model_fc = create_fc_model(length)
	##### LOAD MODEL WEIGHTS #####
	filename = '../trained_models/fc_model_weights_length_%d_trained.h5' % length
	model_fc.load_weights(filename)

	print('Creating Stateful Vanilla RNN Model and Loading Trained Weights...')
	model_rnn_stateful = create_rnn_model(length, True, batch_size)
	##### LOAD MODEL WEIGHTS #####
	filename = '../trained_models/rnn_stateful_model_weights_length_%d_trained.h5' % length
	model_rnn_stateful.load_weights(filename)

	print('Creating stateless Vanilla RNN Model and Loading Trained Weights...')
	model_rnn_stateless = create_rnn_model(length, False, batch_size)
	##### LOAD MODEL WEIGHTS #####
	filename = '../trained_models/rnn_stateless_model_weights_length_%d_trained.h5' % length
	model_rnn_stateless.load_weights(filename)

	print('Creating Stateful LSTM Model and Loading Trained Weights...')
	model_lstm_stateful = create_lstm_model(length, True, batch_size)
	##### LOAD MODEL WEIGHTS #####
	filename = '../trained_models/lstm_stateful_model_weights_length_%d_trained.h5' % length
	model_lstm_stateful.load_weights(filename)

	print('Creating stateless LSTM Model and Loading Trained Weights...')
	model_lstm_stateless = create_lstm_model(length, False, batch_size)
	##### LOAD MODEL WEIGHTS #####
	filename = '../trained_models/lstm_stateless_model_weights_length_%d_trained.h5' % length
	model_lstm_stateless.load_weights(filename)

	# Predict 
	print('Predicting')
	##### PREDICT #####
	predicted_fc = model_fc.predict(x_test, batch_size=batch_size)
	##### CALCULATE RMSE #####
	fc_rmse = rmse_2array(y_test, predicted_fc)[0]
	fc_rmse_list.append(fc_rmse)

	##### PREDICT #####
	predicted_rnn_stateful = model_rnn_stateful.predict(two_d_df_to_three_d(x_test), batch_size=batch_size)
	##### CALCULATE RMSE #####
	rnn_stateful_rmse = rmse_2array(y_test, predicted_rnn_stateful)[0]
	rnn_stateful_rmse_list.append(rnn_stateful_rmse)

	##### PREDICT #####
	predicted_rnn_stateless = model_rnn_stateless.predict(two_d_df_to_three_d(x_test), batch_size=batch_size)
	##### CALCULATE RMSE #####
	rnn_stateless_rmse = rmse_2array(y_test, predicted_rnn_stateful)[0]
	rnn_stateless_rmse_list.append(rnn_stateless_rmse)

	##### PREDICT #####
	predicted_lstm_stateful = model_lstm_stateful.predict(two_d_df_to_three_d(x_test), batch_size=batch_size)
	##### CALCULATE RMSE #####
	lstm_stateful_rmse = rmse_2array(y_test, predicted_lstm_stateful)[0]
	lstm_stateful_rmse_list.append(lstm_stateful_rmse)

	##### PREDICT #####
	predicted_lstm_stateless = model_lstm_stateless.predict(two_d_df_to_three_d(x_test), batch_size=batch_size)
	##### CALCULATE RMSE #####
	lstm_stateless_rmse = rmse_2array(y_test, predicted_lstm_stateless)[0]
	lstm_stateless_rmse_list.append(lstm_stateless_rmse)

	# print('tsteps:{}'.format(tsteps))
	print('length:{}'.format(length))
	print('Fully-Connected RMSE:{}'.format( fc_rmse ))
	print('Stateful Vanilla RNN RMSE:{}'.format( rnn_stateful_rmse ))
	print('Stateless Vanilla RNN RMSE:{}'.format( rnn_stateless_rmse ))
	print('Stateful LSTM RMSE:{}'.format( lstm_stateful_rmse ))
	print('Stateless LSTM RMSE:{}'.format( lstm_stateless_rmse ))


# Save your rmse values for different length input sequence models:
# This file should have 5 rows (one row per model) and
# 10 columns (one column per input length).
# 1st row: fully-connected model
# 2nd row: vanilla rnn stateful
# 3rd row: vanilla rnn stateless
# 4th row: lstm stateful
# 5th row: lstm stateless
filename = 'all_models_rmse_values.txt'
##### PREPARE RMSE ARRAY THAT WILL BE WRITTEN INTO FILE #####
rmse_arr = np.array([fc_rmse_list, rnn_stateful_rmse_list])
np.savetxt(filename, rmse_arr, fmt='%.6f', delimiter='\t')

print("#" * 33)
print('Plotting Results')
print("#" * 33)

# plt.figure()
# plt.plot(data_input[0][:100], '.')
# plt.plot(expected_output[0][:100], '-')
# plt.legend(['Input', 'Expected output'])
# plt.title('Input - First 100 data points')

# Plot and save rmse vs Input Length
plt.figure()
plt.plot( np.arange(min_length,max_length+1), fc_rmse_list, c='black', label='FC')
plt.plot( np.arange(min_length,max_length+1), rnn_stateful_rmse_list, c='blue', label='Stateful RNN')
plt.plot( np.arange(min_length,max_length+1), rnn_stateless_rmse_list, c='cyan', label='Stateless RNN')
plt.plot( np.arange(min_length,max_length+1), lstm_stateful_rmse_list, c='red', label='Stateful LSTM')
plt.plot( np.arange(min_length,max_length+1), lstm_stateless_rmse_list, c='magenta', label='Stateless LSTM')
plt.title('RMSE vs Input Length in Test Set')
plt.xlabel('length of input sequences')
plt.ylabel('rmse')
plt.legend()
plt.grid()
# plt.show()
plt.savefig("loss_vs_length_pretrain_5_curves.png")


