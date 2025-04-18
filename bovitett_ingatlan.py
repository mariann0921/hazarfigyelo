import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
#gépi tanulás


# Minden sor egy ház: [méret m², szobák száma, emelet, központtól távolság (km)]
X = np.array([
    [50, 2, 1, 10],
    [60, 2, 2, 8],
    [70, 3, 3, 7],
    [80, 3, 4, 6],
    [90, 4, 5, 5],
    [100, 4, 6, 4],
    [110, 5, 7, 3],
    [120, 5, 8, 2]
])

# Ár (ezer dollár)
y = np.array([150, 170, 200, 230, 260, 290, 320, 350])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

uj_haz = np.array([[85, 3, 2, 6]])
uj_haz_scaled = scaler.transform(uj_haz)
josolt_ar = model.predict(uj_haz_scaled)

print(f"A becsült ár: {josolt_ar[0]:.2f} ezer dollár")

# Jósolt értékek a meglévő (tanító) adatokra
y_pred = model.predict(X_scaled)

plt.figure(figsize=(6, 6))
plt.scatter(y, y_pred)
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')  # átló
plt.xlabel('Valódi árak (ezer $)')
plt.ylabel('Jósolt árak (ezer $)')
plt.title('Szórásdiagram: Valódi vs. jósolt')
plt.grid(True)
plt.show()