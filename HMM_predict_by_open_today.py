import numpy as np
from hmmlearn.hmm import GaussianHMM
import pandas_datareader.data as web
import datetime
import pandas as pd
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import time


class ModelHMM():
    def __init__(self, company, day_start, day_end, n_days_previous, n_states, verbose):
        self.company = company
        self.day_start = day_start
        self.day_end = day_end
        self.n_days_previous = n_days_previous
        self.n_states = n_states
        self.verbose = verbose
        self.print_model = verbose

    def _get_value_by_positions(self, df, start_index, end_index):
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
            l_X, l_dates, l_close_v, l_volume_v, l_high_v, l_open_v, l_low_v = self._get_value_by_positions(df,
                                                                                                            start_index - 1,
                                                                                                            end_index + 1)
            # Truong hop can predict => khong duoc su dung close_v => thay bang uoc luong khac
            return np.column_stack([(open_v - [l_close_v[0]]) / l_close_v[0], (high_v - open_v) / open_v,
                                    (low_v - open_v) / open_v]), dates, close_v, volume_v, high_v, open_v, low_v
        else:
            # Truong hop tao data cho training
            return np.column_stack([(close_v - open_v) / open_v, (high_v - open_v) / open_v,
                                    (low_v - open_v) / open_v]), dates, close_v, volume_v, high_v, open_v, low_v

    def _show_plot(self, dates, close_v, hidden_states, title):
        years = YearLocator()  # every year
        months = MonthLocator()  # every month
        yearsFmt = DateFormatter('%Y/%m')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.canvas.set_window_title(title)

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

        error = np.sum(np.abs((close_v - hidden_states) * 1.0 / close_v)) * 100.0 / close_v.shape[0]
        return error

    def predict(self):
        self._predict(self.company, self.day_start, self.day_end, self.n_days_previous, self.n_states)

    def _predict(self, company, day_start, day_end, n_previous, n_cluster):
        df = web.DataReader(company, 'google', day_start, day_end)
        n_days = df.shape[0]

        v_X, v_dates, v_close_v, v_volume_v, v_high_v, v_open_v, v_low_v = self._get_value_by_positions(df, 0, n_days)
        predicted = []
        counting_error = 0

        for i in range(n_previous, n_days):
            model = GaussianHMM(n_components=n_cluster, covariance_type="diag", n_iter=5, verbose=False,
                                init_params='mtsc')
            X, dates, close_v, volume_v, high_v, open_v, low_v = self._get_value_by_positions(df, i - n_previous, i)
            if (self.verbose == True):
                print "Predicting in", i - n_previous + 1, "th/", n_days - n_previous + 1, "days..."

            try:
                temp_model = model.fit(X)

                if (self.print_model == True):
                    print "Transform matrix : "
                    print temp_model.transmat_
                    print "Starting probability : "
                    print temp_model.startprob_
                    self.print_model = False

                X, dates, close_v, volume_v, high_v, open_v, low_v = self._get_value_by_positions(df, i, i + 1)
                hidden_states = temp_model.predict(X)
                predicted.append(temp_model.means_[hidden_states[0]][0] * open_v[0] + open_v[0])
            except:
                counting_error += 1
                print(counting_error)
                X, dates, close_v, volume_v, high_v, open_v, low_v = self._get_value_by_positions(df, i, i + 1)
                predicted.append(open_v[0])

        print "Finished predicting", n_days - n_previous + 1, "days in ", time.time() - start_time, " s"
        error = self._show_plot(v_dates[n_previous:], v_close_v[n_previous:], predicted, 'Trained data')
        print "Mean absolute percentage error MAPE = ", error, '%'


####Running#########
start_time = time.time()
day_start = datetime.datetime(2016, 1, 1)
day_end = pd.datetime.today()

model = ModelHMM(company="AAPL", day_start=day_start, day_end=day_end, n_days_previous=100, n_states=10, verbose=True)
model.predict()
