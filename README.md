# HiddenMarkovModelPredictingStock
Predicting stock price by using Hidden Markov model

# Requirement
- Python 
- Library: numpy, Ipython notebook, matplotlib, pandas_datareader, pandas

# Instruction
- HMM_predict_by_open_today.py: Hidden Markov Model using open price of a day to predict close price in days
  Options:
  model = ModelHMM(company="AAPL", day_start=day_start, day_end=day_end, n_days_previous=200, n_states=10, verbose=True, n_decimals = 3, latex = True)
  
  + company    		: Stock name of the company
  + day_start  		: Starting day
  + day_end    		: Ending day
  + n_days_previous	: Number of previous days will be used for training
  + n_states            : Number of hidden states
  + verbose             : Printing when running or not
  + n_decimals          : Number of decimals of double  
  + latex               : Printing matrix in latex type or not

- HMM_predict_continous_days.py: Hidden Markov Model using close price of a day to predict close price n_days_predict days after
  model = ModelHMM(company="AAPL", day_start=day_start, day_end=day_end, n_days_previous=100, n_states=10,
           		n_days_predict=2, verbose=True, n_decimals = 3, latex = False)

  + n_days_predict      : Number of following days will be predict using close price of day
