import numpy as np
import torch
from scipy.stats import norm
import matplotlib.pyplot as plt

from Lstm.DataPreprocess import preprocess_data
from Lstm.LstmModel import LstmRNN
from Lstm.Predict import predict, calculate_diff
from Lstm.Train import train


def gaussian_distribution(predictions, true_values):
    errors = []
    for i in range(len(predictions)):
        errors.append(predictions[i] - true_values[i])
    return errors, norm.fit(errors)


def get_pdf(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    # print(pdf)
    return pdf


def anormaly_score(predictions, true_values):
    errors, (mean, sigma) = gaussian_distribution(predictions, true_values)
    print(mean, sigma)
    return get_pdf(errors, mean, sigma)


def anormaly_points(pdf, threshold, test_data):
    anormaly_x = []
    anormaly_y = []
    for i in range(len(pdf)):
        if pdf[i] < threshold:
            anormaly_x.append(i)
            anormaly_y.append(test_data[i])
    return anormaly_x, anormaly_y


csv_path = 'D:/codes/Python/LSTM/dataset/pollution.csv'
SAVE_PATH = "parameters/"
HIDDEN_SIZE = 200
LAYER = 1
DROP_OUT = 0.3
WINDOW_SIZE = 50
TRAIN_LEFT_BOUND = 10000
TRAIN_RIGHT_BOUND = 15000
TEST_LEFT_BOUND = 20000
TEST_RIGHT_BOUND = 21000
EPOCHS = 50
LEARNING_RATE = 0.001
GRAD_CLIP = 0.5
# TRAINED = False


TRAINED = True

THRESHOLD = 0.004


def main():
    data = preprocess_data(csv_path)
    # show_data_by_plot(data)
    train_data = data['pollution'][TRAIN_LEFT_BOUND:TRAIN_RIGHT_BOUND]
    test_data = data['pollution'][TEST_LEFT_BOUND:TEST_RIGHT_BOUND]
    # model1 = LstmRNN(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=LAYER, dropout=DROP_OUT)
    # model_id = "H" + str(HIDDEN_SIZE) + "-L" + str(LAYER) + "-D" + str(DROP_OUT * 10) + "-W" + str(
    #     WINDOW_SIZE) + "-T" + str(TRAIN_LEFT_BOUND) + "~" + str(TRAIN_RIGHT_BOUND) + "-E" + str(EPOCHS) + "&" + str(
    #     LEARNING_RATE)
    # if not TRAINED:
    #     # method 'train' will return the state_dict with the lowest total_loss
    #     best_state = train(model, train_data, WINDOW_SIZE, epoch=EPOCHS, learning_rate=LEARNING_RATE)
    #     # save_name = input("save as:")
    #     torch.save(best_state, SAVE_PATH + model_id + ".pkl")
    # else:
    #     # open_name = input("load model file:")
    #     model.load_state_dict(torch.load(SAVE_PATH + model_id + ".pkl", map_location=torch.device('cpu')))
    model1 = LstmRNN(input_size=1, hidden_size=200, num_layers=1, dropout=0.3)
    model1.load_state_dict(
        torch.load(SAVE_PATH + "H200-L1-D3.0-W50-T10000~15000-E50&0.001.pkl", map_location=torch.device('cpu')))
    model2 = LstmRNN(input_size=1, hidden_size=200, num_layers=1, dropout=0.1)
    model2.load_state_dict(
        torch.load(SAVE_PATH + "H200-L1-D1.0-W10-T15235~15275-E1200&0.01.pkl")
    )
    predictions1 = predict(model1, test_data, window_size=WINDOW_SIZE)
    predictions2 = predict(model2, test_data, window_size=10)
    # predictions = process_predictions(predictions)
    print("====total diff:{}".format(calculate_diff(
        predictions1, test_data
    )))
    # print(len(test_data))
    # print(len(predictions))
    score = anormaly_score(predictions1, test_data.values)
    anormaly_x, anormaly_y = anormaly_points(score, THRESHOLD, test_data)
    # print(score)
    plt.plot(test_data.values, c='b')
    # plt.plot(predictions, c='y')
    plt.scatter(anormaly_x, anormaly_y, c='r')
    # print(test_data.values)
    plt.show()


if __name__ == '__main__':
    main()
