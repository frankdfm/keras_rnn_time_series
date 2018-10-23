from keras import backend
import matplotlib.pyplot as plt
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def rmse_2array(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def model_fc_factory(num_of_time_steps):
    model = Sequential()
    model.add(Dense(20, activation='relu', input_shape=(num_of_time_steps,)))
    model.add(Dense(1))
    # model.compile(loss='mse', optimizer='rmsprop', metrics=[rmse])
    model.compile(loss='mse', optimizer='rmsprop')
    return model


def model_simplernn_factory(num_of_time_steps, stateful, batch_size):
    assert type(stateful) == type(True)
    model = Sequential()
    # model.add(SimpleRNN(20, stateful=stateful, input_shape=(num_of_time_steps, 1),
    model.add(SimpleRNN(20, stateful=stateful, activation='relu',
                        batch_input_shape=[batch_size, num_of_time_steps,1]))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='rmsprop')
    return model


def model_lstm_factory(num_of_time_steps, stateful, batch_size):
    assert type(stateful) == type(True)
    model = Sequential()
    model.add(LSTM(20, stateful=stateful, input_shape=(num_of_time_steps, 1),
                        batch_input_shape=[batch_size, num_of_time_steps,1]))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='rmsprop')
    return model


def load_raw_data():
    noisy_data = np.loadtxt('filter_data/noisy_data.txt', delimiter='\t', dtype=np.float)
    smooth_data = np.loadtxt('filter_data/smooth_data.txt', delimiter='\t', dtype=np.float)
    assert noisy_data.shape == (1000,)
    assert smooth_data.shape == (1000,)
    return pd.DataFrame(noisy_data), pd.DataFrame(smooth_data)


def oneD_ts_to_n_steps(data_input, length):
    total_sample_size = len(data_input)
    for i in range(1, length):
        data_input[i] = data_input[i - 1].shift(-1)
    data_input = data_input[:-length + 1]
    # [t3, t2, t1, t0] is the right order
    data_input = data_input.iloc[:, ::-1]
    print("arrange shape: ", data_input.shape, " expecting: (%d, %d)" % (total_sample_size - length + 1, length))
    assert data_input.shape == (total_sample_size - length + 1, length)
    return data_input


def prepare_data_for_ts_training(one_d_feature, one_d_label, length):
    n_steps_feature = oneD_ts_to_n_steps(one_d_feature, length)  # ready to throw into rnn
    one_d_label_from_length = one_d_label[length-1:]  # data in the front are lost
    assert n_steps_feature.shape[0] == one_d_label_from_length.shape[0]
    assert n_steps_feature.shape[1] == length
    return n_steps_feature, one_d_label_from_length


def save_loss_vs_iter_plot(train_loss, test_loss, name):
    plt.figure()
    plt.title(name)
    plt.plot(train_loss, label='train_loss')
    plt.plot(test_loss, label='test_loss')
    plt.legend()
    plt.savefig(name)
    plt.clf()


def save_loss_vs_iter_plot_with_name(history, length, plot_name):
    plt.figure()
    plt.plot(history.history['loss'], label='length %s - train_loss' % length)
    # plt.plot(history.history['val_loss'], label='length %s - val_loss' % length)
    plt.legend()
    plt.savefig(plot_name)
    plt.clf()


def two_d_df_to_three_d(a):
    return a.values.reshape(a.shape[0], a.shape[1], 1)



if __name__ == "__main__":
    print(rmse_2array(np.array([1,200,3,4,5]), np.array([0.33332,3,4,5,6])) )

    # length = 2
    # model_fc = model_fc_factory(length)
    # model_fc.load_weights('trained_models/fc_model_weights_length_%d_trained.h5' %length)
    # predicted_fc = model_fc.predict(x_test)

    a, b = load_raw_data()
    a,b = prepare_data_for_ts_training(a,b,3)
    print(a.values[:5])
    # length = 2
    # stateful = True
    # batch_size = 1
    # model_rnn_stateful = model_simplernn_factory(length, stateful, batch_size)
    # filename = 'trained_models/rnn_stateful_model_weights_length_%d_trained.h5' % length
    # model_rnn_stateful.load_weights(filename)
    # predicted_rnn_stateful = model_rnn_stateful.predict(two_d_df_to_three_d(a), batch_size=1)


    # lista = [1,2,3,4]
    # listb = [2,3,4,5]
    # print(np.array([lista,listb]))




    #x = np.random.random((100, 9, 1))
    #y = np.random.random(100)
    #model = Sequential()
    #model.add(SimpleRNN(20, stateful=True, input_shape=(9, 1),
    #                    batch_input_shape=[1, 9, 1]))
    #model.add(Dense(1))
    #model.compile(loss='mse', optimizer='rmsprop')
    #print(x.shape)
    #model.fit(x,y,epochs=1,batch_size=1)

    print("end")
