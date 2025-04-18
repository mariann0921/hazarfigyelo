import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Négyzetméterek (X) és árak (y) ezer dollárban
X = np.array([[50], [60], [70], [80], [90], [100], [110], [120]])  # m²
y = np.array([150, 180, 210, 240, 270, 300, 330, 360])  # ezer dollár

model = LinearRegression()
model.fit(X, y)  # itt megtanulja az összefüggést

#jóslás
uj_haz = np.array([[95]])  # egy 95 m²-es ház
josolt_ar = model.predict(uj_haz)
print(f"A 95 m²-es ház becsült ára: {josolt_ar[0]:.2f} ezer dollár")

# Ábrázolás
plt.scatter(X, y, color='blue', label='Adatok')
plt.plot(X, model.predict(X), color='red', label='Lineáris modell')
plt.xlabel('Méret (m²)')
plt.ylabel('Ár (ezer dollár)')
plt.title('Házár-jóslás gépi tanulással')
plt.legend()
plt.grid(True)
plt.show()