import pickle
import numpy as np

model = pickle.load(open("aqi_model.pkl", "rb"))


def predict_aqi(values):
    values = np.array(values).reshape(1, -1)
    prediction = model.predict(values)
    return prediction[0]


sample = [50, 90, 12, 30, 40, 15, 1.5, 10, 45]

result = predict_aqi(sample)

print("Predicted AQI:", result)