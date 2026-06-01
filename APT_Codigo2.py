# ==========================================================
# EXPERIMENTO 2
# TF-IDF + MULTINOMIAL NAIVE BAYES
# Detección de Edad en Comunicaciones
# Materia: Análisis y Procesamiento Inteligente de Textos
# ==========================================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================================
# CORPUS DE EJEMPLO
# ==========================================================

textos = [
    # Menores
    "bro ya viste el evento nuevo de fortnite",
    "holaaaa como estas jajaja",
    "vamos a jugar minecraft hoy",
    "noooo me mataron otra vez xd",
    "esta bien chido ese skin",
    "jajaja que buena partida",
    "mañana saliendo de clases jugamos",
    "ya viste el nuevo pase de batalla",
    "que cool estuvo la partida",
    "me encanta roblox",

    # Adultos formales
    "adjunto el informe solicitado para su revisión",
    "quedo atento a cualquier comentario",
    "agradezco su apoyo en este proyecto",
    "la reunión se llevará a cabo mañana",
    "es necesario validar la documentación",
    "se realizó el análisis correspondiente",
    "favor de revisar los resultados",
    "adjunto evidencia del proceso",
    "el cliente aprobó la propuesta",
    "se generó el reporte final",

    # Adultos informales
    "jajaja estuvo buena la reunión",
    "te marco más tarde",
    "ando ocupado con el trabajo",
    "vamos por un café después",
    "ya terminé los pendientes",
    "estuvo pesado el día",
    "nos vemos al rato",
    "qué tal estuvo tu día",
    "me avisas cuando llegues",
    "ya salí de la oficina"
]

etiquetas = [
    # Menores
    "Menor", "Menor", "Menor", "Menor", "Menor",
    "Menor", "Menor", "Menor", "Menor", "Menor",

    # Adultos formales
    "Adulto", "Adulto", "Adulto", "Adulto", "Adulto",
    "Adulto", "Adulto", "Adulto", "Adulto", "Adulto",

    # Adultos informales
    "Adulto", "Adulto", "Adulto", "Adulto", "Adulto",
    "Adulto", "Adulto", "Adulto", "Adulto", "Adulto"
]

# ==========================================================
# DIVISIÓN DE DATOS
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    textos,
    etiquetas,
    test_size=0.30,
    random_state=42,
    stratify=etiquetas
)

# ==========================================================
# PIPELINE TF-IDF + NAIVE BAYES
# ==========================================================

modelo = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 3),
            lowercase=True
        )
    ),
    (
        "clasificador",
        MultinomialNB()
    )
])

# ==========================================================
# ENTRENAMIENTO
# ==========================================================

modelo.fit(X_train, y_train)

# ==========================================================
# PREDICCIONES
# ==========================================================

predicciones = modelo.predict(X_test)

# ==========================================================
# MÉTRICAS
# ==========================================================

accuracy = accuracy_score(y_test, predicciones)
precision = precision_score(
    y_test,
    predicciones,
    pos_label="Adulto",
    zero_division=0
)

recall = recall_score(
    y_test,
    predicciones,
    pos_label="Adulto",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predicciones,
    pos_label="Adulto",
    zero_division=0
)

print("\n========== RESULTADOS DEL EXPERIMENTO 2 ==========")
print(f"Accuracy : {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"F1-score : {f1:.2f}")

print("\n========== MATRIZ DE CONFUSIÓN ==========")
print(confusion_matrix(y_test, predicciones))

print("\n========== REPORTE DE CLASIFICACIÓN ==========")
print(classification_report(y_test, predicciones, zero_division=0))

# ==========================================================
# PRUEBA MANUAL INTERACTIVA
# ==========================================================

print("\n========== MODO INTERACTIVO ==========")
print("Escribe un mensaje para detectar el perfil.")
print("Escribe 'salir' para terminar.\n")

while True:
    mensaje = input("Mensaje: ").strip()

    if mensaje.lower() == "salir":
        print("Programa finalizado.")
        break

    if not mensaje:
        print("El mensaje está vacío. Escribe algo para analizar.")
        continue

    resultado = modelo.predict([mensaje])[0]
    print(f"Perfil detectado: {resultado}\n")
