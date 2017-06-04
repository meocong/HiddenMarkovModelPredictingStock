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

    if (abs(start_index - end_index) == 1):
        l_X, l_dates, l_close_v, l_volume_v, l_high_v, l_open_v, l_low_v = get_value_by_positions(df, start_index - 1, end_index + 1)
        # Truong hop can predict => khong duoc su dung close_v => thay bang uoc luong khac
        return np.column_stack([(open_v - [l_close_v[0]]) / l_close_v[0], (high_v - open_v) / open_v,
                                (low_v - open_v) / open_v]), dates, close_v, volume_v, high_v, open_v, low_v
    else:
        # Truong hop tao data cho training
        return np.column_stack([(close_v - open_v)/open_v, (high_v - open_v)/open_v, (low_v - open_v)/open_v]), dates, close_v, volume_v, high_v, open_v, low_v

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
    # ax.fmt_ydata = lambda x: '$%1f' % x
    ax.grid(True)

    fig.autofmt_xdate()
    plt.savefig('plot.png')
    plt.show()

    error = np.sum(np.abs((close_v - hidden_states) * 1.0/close_v))*100.0/close_v.shape[0]
    return error

def Predict(company, start, end, n_previous, n_cluster):
    df = web.DataReader(company, 'google', start, end)
    n_days = df.shape[0]

    v_X, v_dates, v_close_v, v_volume_v, v_high_v, v_open_v, v_low_v = get_value_by_positions(df, 0, n_days)
    predicted = []
    counting_error = 0

    for i in range(n_previous, n_days):
        model = GaussianHMM(n_components=n_cluster, covariance_type="diag", n_iter=100, verbose=False, init_params='mtsc')
        X, dates, close_v, volume_v, high_v, open_v, low_v = get_value_by_positions(df, i-n_previous, i)
        print "Predicting in",i - n_previous + 1,"th/", n_days - n_previous + 1,"days..."

        try:
            temp_model = model.fit(X)
            X, dates, close_v, volume_v, high_v, open_v, low_v = get_value_by_positions(df, i, i+1)
            hidden_states = temp_model.predict(X)
            predicted.append(temp_model.means_[hidden_states[0]][0] * open_v[0] + open_v[0])
        except:
            counting_error += 1
            # print(counting_error)
            X, dates, close_v, volume_v, high_v, open_v, low_v = get_value_by_positions(df, i, i + 1)
            predicted.append(open_v[0])

    print "Finished predicting",n_days - n_previous + 1,"days in ", time.time() - start_time, " s"
    error = show_plot(v_dates[n_previous:], v_close_v[n_previous:], predicted, 'Trained data')
    print "Mean absolute percentage error MAPE = ", error ,'%'


start_time = time.time()
start = datetime.datetime(2016, 1, 1)
end = pd.datetime.today()
Predict("AAPL", start, end, n_previous = 100, n_cluster=10)
