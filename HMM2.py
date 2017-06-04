import numpy as np
from hmmlearn.hmm import GaussianHMM
import pandas_datareader.data as web
import datetime
import pandas as pd
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import time

def get_value_by_positions(df, start_index, end_index):
    X = df.ix[start_index:end_index]
    dates = np.array([q for q in pd.to_datetime(X.reset_index()['Date'], unit='s')])
    close_v = np.array([q for q in X['Close']])
    volume_v = np.array([q for q in X['Volume']])
    high_v = np.array([q for q in X['High']])
    open_v = np.array([q for q in X['Open']])
    low_v = np.array([q for q in X['Low']])

    dates = dates
    close_v = close_v
    volume_v = volume_v
    high_v = high_v
    open_v = open_v
    low_v = low_v
    # return dates, close_v, volume_v, high_v, open_v, low_v
    return np.column_stack([open_v, volume_v, (high_v - open_v)/open_v, (low_v - open_v)/open_v]), dates, close_v, volume_v, high_v, open_v, low_v

def show_plot(dates, close_v, hidden_states,title):
    ###############################################################################
    # print trained parameters and plot
    # print("Transition matrix")
    # print(model.transmat_)
    # print()

    years = YearLocator()  # every year
    months = MonthLocator()  # every month
    yearsFmt = DateFormatter('%Y/%m')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.canvas.set_window_title(title)

    # label_plot = np.zeros(close_v.shape[0])
    # for i in range(model.n_components):
    #     # use fancy indexing to plot data in each state
    #     idx = (hidden_states == i)
    #     # label_plot[idx] = i
    #     ax.plot_date(dates[idx], close_v[idx], '-', label="%dth hidden state" % i)
    # value = [model.means_[x][0] for x in hidden_states]
    value = hidden_states
    ax.plot_date(dates, close_v, '-', label="Actual value")
    ax.plot_date(dates, value, '-', label="Predicted value")

    ax.legend()

    # format the ticks
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.autoscale_view()

    # format the coords message box
    ax.fmt_xdata = DateFormatter('%Y-%m')
    ax.fmt_ydata = lambda x: '$%1f' % x
    ax.grid(True)

    fig.autofmt_xdate()
    plt.savefig('plot.png')
    plt.show()
    return 0

def Predict(company, start, end, n_previous, n_cluster):
    df = web.DataReader(company, 'google', start, end)
    model = GaussianHMM(n_components=n_cluster, covariance_type="tied", n_iter=100, init_params='m', verbose=True)

    v_X, v_dates, v_close_v, v_volume_v, v_high_v, v_open_v, v_low_v = get_value_by_positions(df, 0, df.shape[0])
    predicted = []

    for i in range(n_previous, df.shape[0]):
        X, dates, close_v, volume_v, high_v, open_v, low_v = get_value_by_positions(df, i-n_previous, i)
        print "Predicting in",i,"th day..."

        try:
            temp_model = model.fit(X)
            X, dates, close_v, volume_v, high_v, open_v, low_v = get_value_by_positions(df, i, i+1)
            hidden_states = temp_model.predict(X)
            predicted.append(temp_model.means_[hidden_states[0][0]])
        except:
            X, dates, close_v, volume_v, high_v, open_v, low_v = get_value_by_positions(df, i, i + 1)
            predicted.append(open_v[0])

    print "Finished in ", time.time() - start_time, " s"
    show_plot(v_dates[n_previous:], v_close_v[n_previous:], predicted, 'Trained data')


start_time = time.time()
start = datetime.datetime(2016, 1, 1)
end = pd.datetime.today()
Predict("AAPL", start, end, n_previous = 4, n_cluster=4)
