import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.recurrent import lstm
import pickle
from datetime import date, timedelta

Learning_rate = 1e-3

pickle_in_min = open('TWfuture_min_data_2014_adjusted.pickle', 'rb')
df_min = pickle.load(pickle_in_min)

start_day = date(2014,1,2)
end_day = date(2016,12,31)
interval = timedelta(days=1)

# 15mins -> 15mins
def initTrData(df,start_day,end_day):

    training_data = []
    span_day = end_day-start_day
    span = int(span_day.days)
    last_day = start_day

    for _ in range(span):
        y_price = df[df["date"]==last_day]["close"][-1]
        start_day += interval
        df_today = df[df["date"]==start_day]

        df_today.loc[df_today.index,'open_adjusted'] = (df_today["open"] - y_price)*100/y_price
        df_today.loc[df_today.index,"high_adjusted"] = (df_today["high"] - y_price)*100/y_price
        df_today.loc[df_today.index,"low_adjusted"] = (df_today["low"] - y_price)*100/y_price
        df_today.loc[df_today.index,"close_adjusted"] = (df_today["close"] - y_price)*100/y_price
        df_today.loc[df_today.index,"close_mvag5_adjusted"] = (df_today["close_mvag5"] - y_price)*100/y_price
        df_today.loc[df_today.index,"close_mvag20_adjusted"] = (df_today["close_mvag20"] - y_price)*100/y_price

        if len(df_today) > 0:

            last_day = start_day
            print (start_day)

            for i in range(120):
                df_today_min = df_today[i:i+15][["open_adjusted","high_adjusted", "low_adjusted","close_adjusted","close_mvag5_adjusted","close_mvag20_adjusted","K9","D9"]]
                X = np.array(df_today_min)
                call = False
                put = False
                for j in range(1,16):
                    if (df_today.iloc[i+15+j]["close"] -
                                df_today.iloc[i+6]["open"]) > 10:
                        call = True
                    elif (df_today.iloc[i+15+j]["close"] -
                                df_today.iloc[i+6]["open"]) < -10:
                        put = True
                if call == True and put == False:
                    y = [0,0,1]
                elif call == False and put == True:
                    y = [1,0,0]
                else:
                    y = [0,1,0]
                training_data.append([X,y])

    training_data_save = np.array(training_data)
    np.save('preprocessing_15to15.npy',training_data_save)

initTrData(df_min,start_day,end_day)



