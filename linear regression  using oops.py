import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy as np
import pandas as pd


class LineaRegression:
    def __init__(self, filename):
        self.filename = filename


def read_data(self):
    column_names = ['area', 'price']

    io = pd.read_csv(self.area_price, names=column_names, header=None)
    x_val = (io.values[1:, 0])
    y_val = (io.values[1:, 1])
    size_array = len(y_val)
    for i in range(size_array):
        x_val[i] = float(x_val[i])
        y_val[i] = float(y_val[i])


@staticmethod
def feature_normalize(train_x):
    mean = np.mean(train_x, axis=0)
    std = np.std(train_x, axis=0)
    print(mean, std)
    return (train_x - mean) / std


@staticmethod
def numpy_reg(x_input, y_input):
    w, b = 0.0, 0.0
    num_epoch = 100
    converge_rate = np.zeros([num_epoch, 1], dtype=float)
    learning_rate = 1e-3
    for e in range(num_epoch):
        y_predict = w * x_input + b
        grad_w, grad_b - (y_predicted - y_input).dot(x_input), (y_predicted - y_input).sum()
        return w, b


@staticmethod
def plot_data(x_input, y_input, w, b):
    y_estimated = w * x_input + b
    plt.rcParams.update({'font.size': 20})
    fig = plt.figure()
    fig.set_size_inches(15, 12)
    ax = fig.add_subplot(111)

    ax.scatter(x_input, y_input, color='green', s=70)

    ax.set_xlabel('Number of iterations')

    ax.set_ylabel('Error')
    ax.set_title('Apartment price vs area in Berlin')

    ax.plot(x_input, y_estimated, linewidth=4.0, color='red')

    ax.spines['bottom'].set_color('orange')
    ax.spines['bottom'].set_linewidth(3.0)
    ax.spines['top'].set_color('orange')
    ax.spines['top'].set_linewidth(3.0)
    ax.spines['right'].set_color('orange')
    ax.spines['right'].set_linewidth(3.0)
    ax.spines['left'].set_color('orange')
    ax.spines['left'].set_linewidth(3.0)

    ax.tick_params(axis='x', colors='orange')
    ax.tick_params(axis='y', colors='orange')

    ax.yaxis.label.set_color('orange')
    ax.xaxis.label.set_color('orange')
    ax.title.set_color('orange')
    plt.savefig('numpy_regression_points_norm.png', transparent=True)
    plt.show()
    plt2.rcParams["figure.figsize"] = [10, 6]
    plt2.plot(converge_rate, color='black', linewidth=4)
    plt2.xlabel('Number of iterations')
    plt2.ylabel('Error')
    plt2.title('Convergence')
    plt2.savefig('convergence.png', transparent=True, antialiased=True)
    plt2.show()
