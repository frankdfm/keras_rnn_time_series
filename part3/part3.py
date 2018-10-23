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

# training parameters passed to "model.fit(...)"
batch_size = 1
epochs = 10

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
	
	# Create the stateful model
	print('Creating Stateful LSTM Model...')
	model_lstm_stateful = create_lstm_model(length, True, batch_size)

	# Train the model
	print('Training')
	loss_vs_iter_vals = []
	val_loss_vs_iter_vals = []
	for i in range(epochs):
		print('Epoch', i + 1, '/', epochs)
		# Note that the last state for sample i in a batch will
		# be used as initial state for sample i in the next batch.
		
		##### TRAIN YOUR MODEL #####
		history = model_lstm_stateful.fit(two_d_df_to_three_d(x_train), y_train,
										  validation_data=(two_d_df_to_three_d(x_test), y_test),
										  batch_size=batch_size, epochs=1)

		# record the loss in list manually
		loss_vs_iter_vals.append(history.history['loss'])
		val_loss_vs_iter_vals.append(history.history['val_loss'])

		# reset states at the end of each epoch
		model_lstm_stateful.reset_states()


	# Plot and save loss curves of training and test set vs iteration in the same graph
    ##### PLOT AND SAVE LOSS CURVES #####
	save_loss_vs_iter_plot(loss_vs_iter_vals,
						   val_loss_vs_iter_vals,
						   "lstm_stateful_loss_vs_iter_length_%d.png" % length)

	# Save your model weights with following convention:
	# For example length 1 input sequences model filename
	# lstm_stateful_model_weights_length_1.h5
	##### SAVE MODEL WEIGHTS #####
	filename = 'lstm_stateful_model_weigths_length_%s.h5' % length
	model_lstm_stateful.save_weights(filename)

	# Predict 
	print('Predicting')
	##### PREDICT #####
	predicted_lstm_stateful = model_lstm_stateful.predict(two_d_df_to_three_d(x_test), batch_size=batch_size)

	##### CALCULATE RMSE #####
	lstm_stateful_rmse = rmse_2array(y_test, predicted_lstm_stateful)[0]
	lstm_stateful_rmse_list.append(lstm_stateful_rmse)

	# print('tsteps:{}'.format(tsteps))
	print('length:{}'.format(length))
	print('Stateful LSTM RMSE:{}'.format( lstm_stateful_rmse ))



	####################################################################################################################
	# Create the stateless model
	print('Creating stateless LSTM Model...')
	model_lstm_stateless = create_lstm_model(length, False, batch_size)

	# Train the model
	print('Training')
	##### TRAIN YOUR MODEL #####
	history = model_lstm_stateless.fit(two_d_df_to_three_d(x_train), y_train,
									  validation_data=(two_d_df_to_three_d(x_test), y_test),
									  batch_size=batch_size, epochs=epochs)


	# Plot and save loss curves of training and test set vs iteration in the same graph
	##### PLOT AND SAVE LOSS CURVES #####
	save_loss_vs_iter_plot(history.history['loss'],
						   history.history['val_loss'],
						   "lstm_stateless_loss_vs_iter_length_%d.png" % length)

	# Save your model weights with following convention:
	# For example length 1 input sequences model filename
	# lstm_stateless_model_weights_length_1.h5
	##### SAVE MODEL WEIGHTS #####
	filename = 'lstm_stateless_model_weigths_length_%s.h5' % length
	model_lstm_stateless.save_weights(filename)

	# Predict 
	print('Predicting')
	##### PREDICT #####
	predicted_lstm_stateless = model_lstm_stateless.predict(two_d_df_to_three_d(x_test), batch_size=batch_size)

	##### CALCULATE RMSE #####
	lstm_stateless_rmse = rmse_2array(y_test, predicted_lstm_stateless)[0]
	lstm_stateless_rmse_list.append(lstm_stateless_rmse)

	# print('tsteps:{}'.format(tsteps))
	print('length:{}'.format(length))
	print('Stateless LSTM RMSE:{}'.format( lstm_stateless_rmse ))


# save your rmse values for different length input sequence models - stateful rnn:
filename = 'lstm_stateful_model_rmse_values.txt'
np.savetxt(filename, np.array(lstm_stateful_rmse_list), fmt='%.6f', delimiter='\t')

# save your rmse values for different length input sequence models - stateless rnn:
filename = 'lstm_stateless_model_rmse_values.txt'
np.savetxt(filename, np.array(lstm_stateless_rmse_list), fmt='%.6f', delimiter='\t')

print("#" * 33)
print('Plotting Results')
print("#" * 33)

plt.figure()
plt.plot(data_input[0][:100], '.')
plt.plot(expected_output[0][:100], '-')
plt.legend(['Input', 'Expected output'])
plt.title('Input - First 100 data points')

# Plot and save rmse vs Input Length
plt.figure()
plt.plot( np.arange(min_length,max_length+1), lstm_stateful_rmse_list, c='red', label='Stateful LSTM')
plt.plot( np.arange(min_length,max_length+1), lstm_stateless_rmse_list, c='magenta', label='Stateless LSTM')
plt.title('RMSE vs Input Length in Test Set')
plt.xlabel('length of input sequences')
plt.ylabel('rmse')
plt.legend()
plt.grid()
plt.savefig("loss_vs_length_2_curves.png")


