
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class CarPricePredictor:
    def __init__(self, X, y):
        # Skálázók létrehozása
        self.X_scaler = StandardScaler()
        self.y_scaler = MinMaxScaler()

        # Skálázás
        self.X_scaled = self.X_scaler.fit_transform(X)
        self.y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1))

        # Modell építése
        self.model = Sequential()
        self.model.add(Dense(16, activation='relu', input_shape=(X.shape[1],)))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1))

        self.model.compile(optimizer='adam', loss='mse')

        # Tanítás
        self.history = self.model.fit(self.X_scaled, self.y_scaled, epochs=1000, verbose=0)

    def predict(self, car_features):
        scaled = self.X_scaler.transform(car_features)
        pred_scaled = self.model.predict(scaled)
        pred = self.y_scaler.inverse_transform(pred_scaled)
        return pred[0][0]  

    def plot_predictions(self, y_true):
    
        y_pred_scaled = self.model.predict(self.X_scaled)
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled)

        plt.plot(y_true, label='Valódi árak', marker='o')
        plt.plot(y_pred, label='Jósolt árak', marker='x')
        plt.title('Valódi vs. jósolt autóárak')
        plt.xlabel('Adatminta indexe')
        plt.ylabel('Ár (ezer dollár)')
        plt.legend()
        plt.grid(True)
        plt.show()
#bemeneti adatok
# [évjárat, km_futás, motor_liter, márka_kód]
X = np.array([
    [2010, 180000, 1.2, 0],
    [2012, 150000, 1.4, 1],
    [2014, 130000, 1.6, 0],
    [2016, 100000, 1.8, 1],
    [2018, 80000, 2.0, 0],
    [2020, 60000, 2.0, 1],
    [2021, 40000, 2.0, 0],
    [2022, 20000, 2.0, 1]
])

# Árak ezer dollárban
y = np.array([4.5, 6.0, 7.5, 10.0, 13.0, 15.0, 17.0, 19.0])

# === Modell létrehozása és tanítása ===
predictor = CarPricePredictor(X, y)

# === Új autó jóslása ===
uj_auto = np.array([[2019, 70000, 1.6, 1]])
ar = predictor.predict(uj_auto)
print(f"A becsült ár: {ar:.2f} ezer dollár")

# === Vizualizáció ===
predictor.plot_predictions(y)

