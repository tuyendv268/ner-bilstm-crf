import sys
import matplotlib as plt
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/tuyendv/Desktop/hust/ner-bilstm-crf/")
from src.resources.utils import load_data
import numpy as np

def plot_bar_chart(xAxis, yAxis):
    plt.bar(xAxis,yAxis)
    plt.title('Biểu đồ thống kê độ dài của câu')
    plt.xlabel('Số lượng từ trong câu')
    plt.ylabel('Số lượng câu')
    plt.xticks(rotation=90)
    

def visualize_data():
    train_path = "/home/tuyendv/Desktop/hust/ner-bilstm-crf/src/datas/ner_train_phonlp.txt"
    test_path = "/home/tuyendv/Desktop/hust/ner-bilstm-crf/src/datas/ner_test_phonlp.txt"
    val_path = "/home/tuyendv/Desktop/hust/ner-bilstm-crf/src/datas/ner_valid_phonlp.txt"

    print("-------------loading data----------------")
    train_datas, train_labels = load_data(train_path)
    test_datas, test_labels = load_data(test_path)
    val_datas, val_labels = load_data(val_path)
    print("-------------load data successfull----------------")
    print("number of training sentences: ", len(train_datas))
    print("number of testing sentences: ", len(test_datas))
    print("number of validating sentences: ",len(val_datas))
    datas = train_datas + test_datas + val_datas
    res_dict = {}
    for data in datas:
        if str(len(data)) not in res_dict:
            res_dict[str(len(data))] = 0
        else:
            res_dict[str(len(data))] += 1
    print("res_dict: ", res_dict)
    xAxis, yAxis = [], []
    for keys, values in res_dict.items():
        xAxis.append(int(keys))
        yAxis.append(values) 
    xAxis = np.array(xAxis)
    yAxis = np.array(yAxis)

    index = np.argsort(xAxis)
    xAxis = xAxis[index]
    yAxis = yAxis[index]

    plot_bar_chart(xAxis=xAxis, yAxis=yAxis)
    plt.show()
    return res_dict

if __name__ == "__main__":
    visualize_data()