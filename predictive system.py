import numpy as np
import pickle

# load the saved model
loaded_model = pickle.load(open('D:/JAKY/SEMESTER 4/AI/Machine Learning Projects/Rock vs Mine/trained_model.sav', 'rb'))

input_data = (0.0125,0.0152,0.0218,0.0175,0.0362,0.0696,0.0873,0.0616,0.1252,0.1302,0.0888,0.0500,0.0628,0.1274,0.0801,0.0742,0.2048,0.2950,0.3193,0.4567,0.5959,0.7101,0.8225,0.8425,0.9065,0.9802,1.0000,0.8752,0.7583,0.6616,0.5786,0.5128,0.4776,0.4994,0.5197,0.5071,0.4577,0.3505,0.1845,0.1890,0.1967,0.1041,0.0550,0.0492,0.0622,0.0505,0.0247,0.0219,0.0102,0.0047,0.0019,0.0041,0.0074,0.0030,0.0050,0.0048,0.0017,0.0041,0.0086,0.0058)

# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]=='R'):
  print('The object is a Rock')
else:
  print('The object is a mine')
