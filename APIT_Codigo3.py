# ==========================================================
# EXPERIMENTO 3
# RANDOM FOREST + RASGOS ESTILOMÉTRICOS Y PSICOLINGÜÍSTICOS
# Detección de Edad en Comunicaciones
# ==========================================================

import re
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

textos = [
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
    "Menor","Menor","Menor","Menor","Menor",
    "Menor","Menor","Menor","Menor","Menor",
    "Adulto","Adulto","Adulto","Adulto","Adulto",
    "Adulto","Adulto","Adulto","Adulto","Adulto",
    "Adulto","Adulto","Adulto","Adulto","Adulto",
    "Adulto","Adulto","Adulto","Adulto","Adulto"
]

palabras_juveniles = [
    "bro","xd","jajaja","minecraft","fortnite",
    "roblox","skin","cool","chido","pase",
    "batalla","noooo","holaaaa"
]

palabras_adultas = [
    "informe","revisión","comentario","proyecto",
    "reunión","documentación","análisis","resultados",
    "evidencia","proceso","cliente","propuesta",
    "reporte","trabajo","oficina","pendientes"
]

def extraer_rasgos(texto):
    texto_original = texto
    texto = texto.lower()

    caracteres = len(texto_original)
    palabras = texto.split()

    mayusculas = sum(1 for c in texto_original if c.isupper())
    ratio_mayusculas = mayusculas / caracteres if caracteres > 0 else 0

    caracteres_repetidos = len(re.findall(r"(.)\1{2,}", texto))

    signos = texto_original.count("!") + texto_original.count("?")
    ratio_emojis_signos = signos / caracteres if caracteres > 0 else 0

    longitud_promedio = (
        sum(len(p) for p in palabras) / len(palabras)
        if palabras else 0
    )

    impulsividad = sum(
        1 for palabra in palabras
        if palabra in palabras_juveniles
    )

    tono_analitico = sum(
        1 for palabra in palabras
        if palabra in palabras_adultas
    )

    total_hits = impulsividad + tono_analitico
    ratio_impulsivo = (
        impulsividad / total_hits
        if total_hits > 0 else 0
    )

    return [
        ratio_mayusculas,
        caracteres_repetidos,
        ratio_emojis_signos,
        longitud_promedio,
        impulsividad,
        tono_analitico,
        ratio_impulsivo
    ]

rasgos = [extraer_rasgos(t) for t in textos]

columnas = [
    "ratio_mayusculas",
    "caracteres_repetidos",
    "ratio_emojis_signos",
    "longitud_promedio_palabra",
    "impulsividad_emocional",
    "tono_analitico_laboral",
    "ratio_impulsivo"
]

df = pd.DataFrame(rasgos, columns=columnas)

X_train, X_test, y_train, y_test = train_test_split(
    df,
    etiquetas,
    test_size=0.30,
    random_state=42,
    stratify=etiquetas
)

modelo = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predicciones))
print("Precision:", precision_score(y_test, predicciones, pos_label="Adulto"))
print("Recall:", recall_score(y_test, predicciones, pos_label="Adulto"))
print("F1:", f1_score(y_test, predicciones, pos_label="Adulto"))

print("\nMatriz de Confusión")
print(confusion_matrix(y_test, predicciones))

print("\nReporte")
print(classification_report(y_test, predicciones))

print("\nModo interactivo")
while True:
    mensaje = input("Mensaje ('salir' para terminar): ")

    if mensaje.lower() == "salir":
        break

    datos = pd.DataFrame(
        [extraer_rasgos(mensaje)],
        columns=columnas
    )

    resultado = modelo.predict(datos)[0]
    print("Perfil detectado:", resultado)
