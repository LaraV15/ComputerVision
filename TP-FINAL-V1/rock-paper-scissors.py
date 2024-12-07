import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os

# Cargar el modelo entrenado
model = load_model('rps_model.h5')

# Inicializar MediaPipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Diccionario para convertir predicciones en etiquetas
GESTURE_NAMES = {0: "Piedra", 1: "Papel", 2: "Tijeras"}

# Inicializar la detección de manos
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                    min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("No se pudo acceder a la cámara.")
            break

        # Convertir BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar la imagen para detectar manos
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar los landmarks en la imagen
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Obtener las coordenadas (x, y) solamente
                coordenadas = [(landmark.x, landmark.y) for landmark in hand_landmarks.landmark]

                # Convertir a numpy array y aplanar
                input_data = np.array(coordenadas).flatten()

                # Asegurarse de que la entrada sea de tamaño (1, 42) para el modelo (21 puntos con 2 coordenadas cada uno)
                input_data = np.expand_dims(input_data, axis=0)

                # Predecir el gesto
                prediction = model.predict(input_data)
                class_id = np.argmax(prediction)
                gesture = GESTURE_NAMES[class_id]
                print(prediction)
                # Mostrar el gesto reconocido en la pantalla
                cv2.putText(frame, f"Gesto: {gesture}", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Mostrar el video en tiempo real
        cv2.imshow('Rock-Paper-Scissors', frame)

        # Salir con la tecla 'q'
        if key == ord('q'):
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()