import numpy as np
from hmmlearn.hmm import GaussianHMM
import pandas_datareader.data as web
import datetime
import pandas as pd
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter

def get_value_by_dates(df, datestart, dateend):
    X = df[datestart : dateend]
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

def show_plot(model, dates, close_v, hidden_states,title):
    ###############################################################################
    # print trained parameters and plot
    # print("Transition matrix")
    # print(model.transmat_)
    # print()

    years = YearLocator()  # every year
    months = MonthLocator()  # every month
    yearsFmt = DateFormatter('%Y')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.canvas.set_window_title(title)

    # label_plot = np.zeros(close_v.shape[0])
    # for i in range(model.n_components):
    #     # use fancy indexing to plot data in each state
    #     idx = (hidden_states == i)
    #     # label_plot[idx] = i
    #     ax.plot_date(dates[idx], close_v[idx], '-', label="%dth hidden state" % i)
    value = [model.means_[x][0] for x in hidden_states]
    ax.plot_date(dates, close_v, '-', label="Actual value")
    ax.plot_date(dates, value, '-', label="Predicted value")

    ax.legend()

    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)
    ax.autoscale_view()

    # format the coords message box
    ax.fmt_xdata = DateFormatter('%Y-%m-%d')
    ax.fmt_ydata = lambda x: '$%1.2f' % x
    ax.grid(True)

    fig.autofmt_xdate()
    plt.show()
    return 0

def show_plot_actual(model, dates, close_v, predicted):
    ###############################################################################
    # print trained parameters and plot
    # print("Transition matrix")
    # print(model.transmat_)
    # print()

    years = YearLocator()  # every year
    months = MonthLocator()  # every month
    yearsFmt = DateFormatter('%Y')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for i in range(model.n_components):
    #     # use fancy indexing to plot data in each state
    #     idx = (hidden_states == i)
    #     ax.plot_date(dates[idx], close_v[idx], '--', label="%dth hidden state" % i)
    ax.plot_date(dates, close_v, '--')
    ax.legend()

    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)
    ax.autoscale_view()

    # format the coords message box
    ax.fmt_xdata = DateFormatter('%Y-%m-%d')
    ax.fmt_ydata = lambda x: '$%1.2f' % x
    ax.grid(True)

    fig.autofmt_xdate()
    plt.show()
    return 0

start = datetime.datetime(2013, 1, 1)
end = pd.datetime.today()
df = web.DataReader("GOOGL", 'google', start, end)

datestart = '20130101'
dateend = '20160101'
# dates, close_v, volume_v, high_v, open_v, low_v = get_value_by_dates(df, datestart, dateend)
# X = np.column_stack([close_v, volume_v, high_v, open_v, low_v])
X, dates, close_v, volume_v, high_v, open_v, low_v = get_value_by_dates(df, datestart, dateend)
model = GaussianHMM(n_components=100, covariance_type="tied", n_iter=100, init_params='m', verbose=True).fit(X)
hidden_states = model.predict(X)
print(hidden_states)

# print("Transition matrix")
# print(model.transmat_)
# print()

print("Means and vars of each hidden state")
for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ", model.means_[i])
    print("var = ", np.diag(model.covars_[i]))
    print()

# fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
# colours = cm.rainbow(np.linspace(0, 1, model.n_components))
# for i, (ax, colour) in enumerate(zip(axs, colours)):
#     # Use fancy indexing to plot data in each state.
#     mask = hidden_states == i
#     ax.plot_date(dates[mask], close_v[mask], ".-", c=colour)
#     ax.set_title("{0}th hidden state".format(i))
#
#     # Format the ticks.
#     ax.xaxis.set_major_locator(YearLocator())
#     ax.xaxis.set_minor_locator(MonthLocator())
#
#     ax.grid(True)
#
# plt.show()

show_plot(model, dates, close_v, hidden_states,'Training data')




###################################
# Test
datestart = '20160102'
dateend = '20170101'
data_test, dates, close_v, volume_v, high_v, open_v, low_v = get_value_by_dates(df, datestart, dateend)
predicted = model.predict(data_test)
show_plot(model, dates, close_v, predicted, 'Test data')
print(predicted)
