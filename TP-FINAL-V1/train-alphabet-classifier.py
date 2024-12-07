import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input


# Cargar los datos desde los archivos CSV
train_data = pd.read_csv('train_landmarks.csv')
test_data = pd.read_csv('test_landmarks.csv')

# Separar características (landmarks) y etiquetas
X_train = train_data.iloc[:, :-1].values  # Todas las columnas excepto la última (features)
y_train = train_data.iloc[:, -1].values   # Última columna (labels)

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir las etiquetas a one-hot encoding
# num_classes = len(np.unique(y_train)) 
# y_train = to_categorical(y_train, num_classes=num_classes)
# y_test = to_categorical(y_test, num_classes=num_classes)

# Definir el modelo de Red Neuronal
model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Definir explícitamente la capa de entrada
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Guardar el modelo entrenado
model.save('asl_model.h5')

# Evaluar el modelo en los datos de prueba
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy en datos de prueba: {test_accuracy:.2f}")

# Graficar el accuracy durante el entrenamiento
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
