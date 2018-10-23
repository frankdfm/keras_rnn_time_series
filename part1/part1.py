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
def create_fc_model(num_of_time_steps):
    return model_fc_factory(num_of_time_steps)
    ##### YOUR MODEL GOES HERE #####
    # model = Sequential()
    # model.add(Dense(20, activation='relu', input_shape=(num_of_time_steps,)))
    # model.add(Dense(1))
    # model.compile(loss='mse', optimizer='rmsprop', metrics=[rmse])
    # model.compile(loss=rmse, optimizer='adam')
    # model.compile(loss=rmse, optimizer='rmsprop')
    return model


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
fc_rmse_list = list()

for num_input in range(min_length, max_length + 1):
    length = num_input

    print("*" * 33)
    print("INPUT DIMENSION:{}".format(length))
    print("*" * 33)

    # convert numpy arrays to pandas dataframe
    data_input = pd.DataFrame(noisy_data)
    expected_output = pd.DataFrame(smooth_data)

    # TODO: DELETE ME debug
    # data_input = data_input[:50]
    # expected_output = expected_output[:50]

    # when length > 1, arrange input sequences
    if length > 1:
        data_input, expected_output = prepare_data_for_ts_training(data_input, expected_output, length)
        # total_sample_size = len(data_input)
        # ##### ARRANGE YOUR DATA SEQUENCES #####
        # for i in range(1, length):
        #     data_input[i] = data_input[i-1].shift(-1)
        # data_input = data_input[:-length+1]
        # expected_output = expected_output[:-length+1]
        # print("arrange shape: ", data_input.shape, " expecting: (%d, %d)" % (total_sample_size - length + 1, length))
        # assert data_input.shape == (total_sample_size-length+1, length)
        # assert expected_output.shape == (total_sample_size-length+1, 1)

    print('data_input length:{}'.format(len(data_input.index)))

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

    # Create the model
    print('Creating Fully-Connected Model...')
    model_fc = create_fc_model(length)

    # Train the model
    print('Training')
    ##### TRAIN YOUR MODEL #####
    history = model_fc.fit(x_train, y_train, validation_data=(x_test, y_test) , epochs=epochs)

    # Plot and save loss curves of training and test set vs iteration in the same graph
    ##### PLOT AND SAVE LOSS CURVES #####
    # plt.plot(history.history['loss'], label='length %s - train_loss' % length)
    # plt.plot(history.history['val_loss'], label='length %s - val_loss' % length)
    # plt.legend()
    # plt.savefig("loss_iter_length_%s.png" % length)
    # plt.clf()
    save_loss_vs_iter_plot(history.history['loss'],
                           history.history['val_loss'],
                           "fc_loss_vs_iter_length_%d.png" % length)

    # Save your model weights with following convention:
    # For example length 1 input sequences model filename
    # fc_model_weights_length_1.h5
    ##### SAVE MODEL WEIGHTS #####
    filename = 'fc_model_weights_length_%s.h5' % length
    model_fc.save_weights(filename)

    # Predict
    print('Predicting/evaluating')
    ##### PREDICT #####
    # predicted_fc = model_fc.predict(x_test)
    predicted_fc = model_fc.predict(x_test, batch_size=batch_size)
    # evaluate_score = model_fc.evaluate(x=x_test, y=y_test)

    ##### CALCULATE RMSE #####
    fc_rmse = rmse_2array(y_test, predicted_fc)[0]
    fc_rmse_list.append(fc_rmse)

    # print('tsteps:{}'.format(tsteps))
    print('length:{}'.format(length))
    print('Fully-Connected RMSE:{}'.format(fc_rmse))


# save your rmse values for different length input sequence models:
filename = 'fc_model_rmse_values.txt'
np.savetxt(filename, np.array(fc_rmse_list), fmt='%.6f', delimiter='\t')

print("#" * 33)
print('Plotting Results')
print("#" * 33)

# Plot and save rmse vs Input Length
plt.figure()
plt.plot(np.arange(min_length, max_length + 1), fc_rmse_list, c='black', label='FC')
plt.title('RMSE vs Input Length in Test Set')
plt.xlabel('length of input sequences')
plt.ylabel('rmse')
plt.legend()
plt.grid()
plt.savefig("rmse_vs_length.png")
# plt.show()

