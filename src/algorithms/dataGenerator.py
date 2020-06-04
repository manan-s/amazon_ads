'''
This file generates synthetic data for the Anomaly Detection System.
Stock Price Data from IBM is used here.
Only used for testing purposes

Output file: data.json

Output file Format:
{
    "ASIN0": [
        {
            "Current_Price": 112.69646453857422,
            "TimeStamp": "2016-01-01"
        },
        {
            "Current_Price": 111.32891082763672,
            "TimeStamp": "2016-01-02"
        },
        ....
    ]
}
'''
import json
import yfinance as yf
import numpy as np
from datetime import timedelta, date

numProducts = 1
data = {}

for i in range(numProducts):
    ProductKey = 'ASIN'+str(i) 
    data[ProductKey] = []

    RawData = yf.download('IBM','2016-01-01','2016-04-30')
    RawData = np.array(list(RawData['Adj Close']))
    RawDataSize = len(RawData)

    StartDate = date(2016, 1, 1)
    DateList = [StartDate + timedelta(n) for n in range(RawDataSize)]
    
    for j in range(RawDataSize):
        DataPoint = {}
        DataPoint['Current_Price'] = RawData[j]
        DataPoint['TimeStamp'] = str(DateList[j])

        data[ProductKey].append(DataPoint)

File = open('data.json', 'w')
out = json.dumps(data, indent=4)
File.write(out)