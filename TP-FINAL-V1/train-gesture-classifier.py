import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle  # Importar shuffle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Cargar los datos del archivo .npy
dataset = np.load('rps_dataset.npy', allow_pickle=True)

# Separar las coordenadas x, y y aplanarlas
coordinates = np.array([np.array(row[2]).flatten() for row in dataset])
labels = np.array([row[1] for row in dataset])

# Preprocesar los datos, se convierten las etiquetas en una matriz de one-hot
X = coordinates
y = to_categorical(labels, num_classes=3)

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir la Red Neuronal
model = Sequential([
    Dense(16, activation='relu', input_shape=(42,)),
    #Dropout(0.5),
    Dense(8, activation='relu'),
    #Dropout(0.5),
    Dense(3, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Modelo guardado en 'rps_model.h5'.")

#Entrenar el modelo con los datos de entrenamiento, separando un 20% de ellos para validación
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

# Guardar el modelo en un archivo .h5
model.save('rps_model.h5')

# Evaluar el modelo
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy en datos de prueba: {test_accuracy:.2f}")

# Imprimir gráfico con valores de accuracy durante el entrenamiento
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.show()