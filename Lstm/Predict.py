import torch

from Lstm.DataPreprocess import preprocess_data, show_data_by_plot
from Lstm.LstmModel import LstmRNN
from Lstm.Train import train
from matplotlib import pyplot as plt


# csv_path = 'D:/codes/Python/LSTM/dataset/pollution.csv'
# SAVE_PATH = "parameters/"
# HIDDEN_SIZE = 200
# LAYER = 1
# DROP_OUT = 0.1
# WINDOW_SIZE = 10
# TRAIN_LEFT_BOUND = 15235
# TRAIN_RIGHT_BOUND = 16275
# TEST_LEFT_BOUND = 20000
# TEST_RIGHT_BOUND = 21000
# EPOCHS = 10
# LEARNING_RATE = 0.01
# GRAD_CLIP = 0.5
# TRAINED = False
#
#
# # TRAINED = True


# 用来差分数据
def data_difference(data):
    for i in range(len(data) - 1, 0, -1):
        data[i] = data[i] - data[i - 1]
    return data


def recover_dif_data(data):
    for i in range(1, len(data) - 1):
        data[i] = data[i - 1] + data[i]
    return data


def predict(model, data, window_size=50, predict_size=1):
    data = torch.tensor(data.values).float()
    # print(input.shape)
    predictions = []
    past_hidden = model.init_hidden(1)
    predict_input = data[0:window_size].view(window_size, 1, 1)
    for i in range(len(data) - window_size):
        pred, hidden = model(predict_input, past_hidden)
        predictions.append(pred[-1, 0, 0].detach().numpy().ravel()[0])
        # print(predict_input.shape)
        past_hidden = model.init_hidden(1)
        # 将下一个窗口的点作为输入
        predict_input = data[i + 1: i + window_size + 1].view(-1, 1, 1)
    print(predictions)
    return data[0:window_size].tolist() + predictions


def calculate_diff(predictions, true_values):
    total_diff = 0
    for i in range(len(predictions)):
        total_diff += abs(predictions[i] - true_values[i])
    return total_diff

# def process_predictions(predictions):
#     dict = {}
#     for predict in predictions:
#         if predict in dict:
#             dict[predict] += 1
#         else:
#             dict[predict] = 1
#     maximum_predict_num = max(dict, key=dict.get)
#     for i in range(len(predictions)):
#         if predictions[i] == maximum_predict_num:
#             left_value = right_value = maximum_predict_num
#             for j in range(i, -1, -1):
#                 if predictions[j] != maximum_predict_num:
#                     left_value = predictions[j]
#                     break
#             for k in range(i, len(predictions)):
#                 if predictions[k] != maximum_predict_num:
#                     right_value = predictions[k]
#                     break
#             predictions[i] = (left_value + right_value) / 2
#     return predictions


# def main():
#     data = preprocess_data(csv_path)
#     # show_data_by_plot(data)
#     train_data = data['pollution'][TRAIN_LEFT_BOUND:TRAIN_RIGHT_BOUND]
#     test_data = data['pollution'][TEST_LEFT_BOUND:TEST_RIGHT_BOUND]
#     model = LstmRNN(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=LAYER, dropout=DROP_OUT)
#     model_id = "H" + str(HIDDEN_SIZE) + "-L" + str(LAYER) + "-D" + str(DROP_OUT * 10) + "-W" + str(
#         WINDOW_SIZE) + "-T" + str(TRAIN_LEFT_BOUND) + "~" + str(TRAIN_RIGHT_BOUND) + "-E" + str(EPOCHS) + "&" + str(
#         LEARNING_RATE)
#     if not TRAINED:
#         # method 'train' will return the state_dict with the lowest total_loss
#         best_state = train(model, train_data, WINDOW_SIZE, epoch=EPOCHS, learning_rate=LEARNING_RATE)
#         # save_name = input("save as:")
#         torch.save(best_state, SAVE_PATH + model_id + ".pkl")
#     else:
#         # open_name = input("load model file:")
#         model.load_state_dict(torch.load(SAVE_PATH + model_id + ".pkl", map_location=torch.device('cpu')))
#     predictions = predict(model, test_data)
#     predictions = test_data[0:WINDOW_SIZE].tolist() + predictions
#     # predictions = process_predictions(predictions)
#     print("====total diff:{}".format(calculate_diff(
#         predictions, test_data
#     )))
#     # print(len(test_data))
#     # print(len(predictions))
#     # dict is used to calculate nums of appearance
#     plt.plot(test_data.values, c='b')
#     plt.plot(predictions, c='y')
#     # print(test_data.values)
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()
