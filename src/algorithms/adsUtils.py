import pickle
import json

def create_pickle(json_file_path):
    """
    Creates a pickle file from the JSON data

    Args
        json_file_name: STRING, The path of the JSON file
        
    Returns
        A string that is the file name of pickle file created
    """

    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    data = {}
    data['timestamp'] = []
    data['value'] = []

    queryASIN = list(json_data.keys())[0]
    dataList = json_data[queryASIN]
    numPoints = len(dataList)

    for count in range(numPoints):
        tempDict = dataList[count]
        timestamp = tempDict["TimeStamp"]
        price = tempDict["Current_Price"]
        data['timestamp'].append(timestamp)
        data['value'].append(price)

    pickle_file_name = 'data.pkl'

    with open(pickle_file_name, 'wb') as f:
        pickle.dump(data, f)

    return pickle_file_name
