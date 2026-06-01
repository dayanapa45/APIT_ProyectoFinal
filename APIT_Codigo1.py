import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# =====================================================================
# 1. LEXICÓN PSICOLINGÜÍSTICO EN ESPAÑOL (equivalente a LIWC/Empath)
# =====================================================================
LEXICON_ES = {
    "emocion_negativa": [
        "odio","odias","asco","horrible","fatal","pesimo","terrible","maldito",
        "estupido","idiota","tonto","inutil","basura","porqueria","ugh","grr",
        "no se vale","injusto","injusticia","me enoja","me molesta","me cae mal",
        "me tiene harto","me tiene harta","que rabia","que ira","que coraje",
    ],
    "emocion_positiva": [
        "feliz","felizz","felizzz","amo","adoro","genial","increible","wow",
        "wiii","yay","yeee","me encanta","que bueno","que rico","esta bueno",
        "ta bueno","ta buenisimo","lo mejor","crack","pro","epico","amazing",
    ],
    "impulsividad": [
        "omg","wtf","lol","xd","jaja","jeje","jiji","aaaa","aaaaa","nooo",
        "siii","pleaseee","porfaaa","porfavor","ayuda","auxilio","socorro",
        "noooo","yaaaas","wooow","uff","ufff","ayyy","ay no","no manches",
    ],
    "exaltacion": [
        "te juro","te lo juro","en serio","literal","literalmente","obvio",
        "obviamente","claramente","re mala","re buena","re toxic","re facil",
    ],
    "temas_juveniles": [
        "profe","escuela","tarea","examen","salon","recreo","clases","maestro",
        "maestra","roblox","minecraft","fortnite","among us","valorant","warzone",
        "discord","twitch","tiktok","server","gaming","mis papas","mi mama",
        "mi papa","mis jefes","me regañaron","me castigaron","me quitaron",
        "no me dejan","no me dejaron","crush",
    ],
    "analitico": [
        "análisis","analizar","analisis","considero","propongo","evaluamos",
        "revisamos","revisaré","verificar","identificar","determinar","concluir",
        "según","conforme","respecto","en relación","en cuanto","por lo tanto",
        "sin embargo","no obstante","asimismo","además","es importante",
        "es fundamental","es indispensable","cabe mencionar",
    ],
    "laboral": [
        "proyecto","contrato","reunión","junta","cliente","proveedor","oficina",
        "informe","reporte","presupuesto","propuesta","empresa","organización",
        "departamento","área","equipo","colega","directivo","gerente","director",
        "factura","pago","inversión","auditoría","normativa","cláusula","acuerdo",
        "agenda","minuta","acta","plazo","entrega","cierre",
    ],
    "cortesia_adulta": [
        "estimado","estimados","saludos","cordialmente","atentamente","quedo",
        "agradezco","agradecer","disculpe","perdón la tardanza","con gusto",
        "a sus órdenes","me complace","adjunto","remito","comparto","le informo",
        "les informo","quedo a sus órdenes","en espera de",
    ],
}

def extraer_rasgos_psicoling(texto):
    texto_lower = texto.lower()
    score_imp = sum(
        1 for cat in ["emocion_negativa","emocion_positiva","impulsividad","exaltacion","temas_juveniles"]
        for term in LEXICON_ES[cat] if term in texto_lower
    )
    score_anal = sum(
        1 for cat in ["analitico","laboral","cortesia_adulta"]
        for term in LEXICON_ES[cat] if term in texto_lower
    )
    ratio = score_imp / (score_imp + score_anal + 1e-5)
    return score_imp, score_anal, ratio


# =====================================================================
# 2. EXTRACTOR DE CARACTERÍSTICAS COMBINADAS
# =====================================================================
FEATURE_COLS = [
    'caps_ratio', 'caracteres_repetidos', 'emojis_y_signos', 'long_promedio_palabra',
    'psicoling_impulsividad', 'psicoling_analitico', 'psicoling_ratio_impulsivo'
]

def calcular_metricas_estilo(textos):
    features = []
    for texto in textos:
        texto_str = str(texto)
        total_caracteres = len(texto_str) if len(texto_str) > 0 else 1
        caps_ratio           = sum(1 for c in texto_str if c.isupper()) / total_caracteres
        caracteres_repetidos = len(re.findall(r'(.)\1{2,}', texto_str))
        emojis_y_signos      = len(re.findall('[\u2600-\U0001f9ff]|[!?¡¿]', texto_str)) / total_caracteres
        palabras             = texto_str.split()
        long_prom            = np.mean([len(p) for p in palabras]) if palabras else 0
        imp, anal, ratio     = extraer_rasgos_psicoling(texto_str)
        features.append([caps_ratio, caracteres_repetidos, emojis_y_signos, long_prom, imp, anal, ratio])
    return pd.DataFrame(features, columns=FEATURE_COLS)


# =====================================================================
# 3. DATASET DE ENTRENAMIENTO
# =====================================================================
textos_menores = [
    "holaaaaa de nuevo!! juegas roblox? agregame soy pro 😎🎮",
    "omg me hackearon la cuenta!! q mal servicio no se vale un baneo 😭🤬",
    "eres un bot literal, no sabes ni jugar gears. manquísimo biba minecraft xd",
    "vayan a mi canal, subí nuevo video de fortnite pasen id para jugar ya",
    "jajajaja q risa weee, el profe de la escuela nos dejo un buen de tarea",
    "pasa tu discord we para jugar halo mañana temprano despues de clases !!",
    "nooo me expulsaron del server 😭😭 q injusto la mod es re mala onda",
    "alguien tiene robux gratis?? necesito skins nuevas para mi personaje xd",
    "oye agrega me al grupo de whats del salon please 🙏🙏",
    "jeje hoy no fui a la escuela y me la pase jugando todo el dia wiii 🎉",
    "me compraron una ps5 mis papas!! to felizzzz 😍😍😍",
    "quien quiere jugar among us esta noche?? yo soy el mejor impostor lol",
    "uwu hola a todos soy nueva aqui espero hacer amigos 🌸💖",
    "broo el profe me bajo puntos por nada, to enojado 🤬🤬",
    "shakira saco cancion nueva alguien ya la escucho?? ta buenisimaaaaa",
    "mi mama no me deja salir el finde, to triste 😢 ñoña",
    "hize trampa en el examen y casi me cacha el profe jajaja que susto!!",
    "wey me cae mal ese tipo del server es re toxic reportenlo!!",
    "alguien juega valorant?? busco duo para rankeds soy gold 2 uwu",
    "tarea de matematicas es imposible alguien me pasa las respuestas porfaaa 🥺",
    "jaja q oso me cacharon copiando en el examen 😳",
    "jaja no manches se me olvidó la tarea otra vez 😂",
    "jaja casi me cacha mi profe con el cel en clases xd",
    "q pereza ir a la escuela hoy mejor me quedo 😴😴",
    "no sé nada del examen de mañana, igual ni estudié lol",
    "mi crush me habló hoy jaja no supe ni que decir 🥺🥺",
    "me corrieron del server por hacer trampa jaja era obvio xd 😂",
    "oye me prestas la tarea de bio? es que no pude 🙈",
    "bro acabo de subir 10 kills seguidas en warzone jaja soy crack 😎",
    "mis papas me quitaron el cel de castigo ugh odio esto 😤",
]
textos_adultos_formales = [
    "Saludos a todos. El día de mañana presentaré el informe financiero de la organización.",
    "Considero preocupante el nivel de exposición de los menores en la red actual.",
    "Agradezco su retroalimentación respecto a la vacante técnica de desarrollo de software.",
    "Es indispensable verificar la identidad del usuario para mitigar riesgos legales en sistemas.",
    "Estaba revisando la agenda académica y me parece adecuado el marco teórico propuesto.",
    "Estimados colegas, comparto el enlace de la sesión informativa programada para las 15:00 horas.",
    "Adjunto el contrato revisado con las cláusulas actualizadas según lo acordado en la reunión.",
    "Les recuerdo que el plazo para la entrega de reportes vence el próximo viernes a las 18:00.",
    "Tras analizar las métricas del trimestre, propongo una reestructuración del área de ventas.",
    "La auditoría interna detectó inconsistencias en el registro contable del segundo semestre.",
    "Quedo a sus órdenes para coordinar los detalles logísticos del evento corporativo.",
    "Es fundamental mantener la confidencialidad de los datos de los clientes según la normativa.",
    "Solicito amablemente que revisen el presupuesto antes de la junta directiva del lunes.",
    "Según el análisis de riesgo, debemos diversificar la cartera de inversiones este año.",
    "Me complace informar que el proyecto fue aprobado por el comité ejecutivo sin observaciones.",
    "Confirmo mi asistencia a la conferencia de innovación tecnológica del próximo mes.",
    "El departamento jurídico revisó el convenio y recomienda ajustar la cláusula de rescisión.",
    "Agradezco la oportunidad de participar en este foro académico de alto nivel.",
    "Nos encontramos evaluando distintos proveedores para renovar la infraestructura de servidores.",
    "Remito el acta de la sesión anterior para su validación y firma por parte de los directivos.",
]
textos_adultos_informales = [
    "Hola amigo, perdón que apenas te contesté ayer andaba en el centro buscando unas cosas para uno de mis proyectos y tremendo desastre que está, me hice 3 horas de regreso y Lit ya nada más llegué y me mimi jaja 🫠",
    "ay no, se me fue el tiempo en el trabajo y ya ni cené, mañana sin falta lo arreglamos jaja 😅",
    "oye perdona la tardanza, estaba en una junta que se extendió más de lo planeado, ya te marco",
    "no manches acabo de llegar del súper y se me olvidó lo más importante, mañana voy de nuevo 😂",
    "ya llegué al depa, fue un día larguísimo pero productivo, mañana te cuento todo en la oficina",
    "jaja sí, ayer fui al médico y me tuvo esperando como 2 horas, al final todo bien afortunadamente",
    "oye te mando los archivos del proyecto ahorita, estaba en el coche cuando me llegó tu mensaje",
    "disculpa no te respondí antes, tenía una llamada con un cliente que se alargó bastante 😅",
    "acabo de salir del gimnasio, qué flojera pero ya me siento mejor jaja, ¿mañana comemos?",
    "se me fue la onda con el pago de la tarjeta, ya lo hice ahorita, qué desastre el mes jaja 🙈",
    "buen día! ayer se cortó la llamada, retomamos hoy en la tarde si puedes, saludos",
    "sí ya vi tu mensaje, estaba manejando, en un rato te llamo para afinar los detalles del contrato",
    "oye me avisas cuando llegues al restaurante, yo salgo de la oficina como a las 7 🙂",
    "jaja no te preocupes, a todos nos pasa, reagendamos la reunión para el jueves sin problema",
    "acabo de terminar de revisar tu propuesta, tiene muy buena pinta, hablamos mañana con calma",
    "jaja fue un día agotador, llegué tardísimo del trabajo y ya no pude ni cenar 😅",
    "oye perdona, estaba en una reunión, ahorita te marco para lo del contrato",
    "no pude ir al banco hoy, estaba con reuniones todo el día, mañana sin falta lo resuelvo",
    "sí ya vi tu correo, déjame revisarlo bien esta noche y mañana te doy retroalimentación 🙂",
    "jaja me agarró el tráfico horrible, llegué una hora tarde a la cita, ya todo bien al final",
    "oye ya revisé los números que me mandaste, hay un par de cosas que platicar, ¿hablamos hoy?",
    "se me fue el día entre juntas y entregas, apenas pude comer algo rápido 😅 mañana charlamos",
    "jaja sí, el fin de semana estuve armando unos muebles nuevos para la oficina en casa, todo un reto",
    "perdón por el retraso en la respuesta, tuve viaje de trabajo toda la semana, ya estoy de vuelta",
    "oye gracias por cubrirme en la reunión, te debo una, la semana que entra te invito el café ☕",
    "acabo de llegar del aeropuerto, viaje agotador pero valió la pena, el cliente quedó muy contento",
    "jaja ya ves, siempre que uno planea algo se complica, al final resolvimos todo bien afortunadamente",
    "no manches se fue la luz en la oficina justo cuando iba a mandar el reporte, qué coraje 😅",
    "ya mandé los documentos por correo, avísame si necesitas algo más para el cierre del proyecto",
    "oye me confirmó el proveedor para el viernes, ¿puedes estar en la llamada a las 11?",
]

todos_textos    = textos_menores + textos_adultos_formales + textos_adultos_informales
todas_etiquetas = [0]*30 + [1]*20 + [1]*30

df_entrenamiento = pd.DataFrame({"texto": todos_textos, "etiqueta": todas_etiquetas})
X_estilo_train   = calcular_metricas_estilo(df_entrenamiento['texto'])
X_train_completo = pd.concat([df_entrenamiento[['texto']], X_estilo_train], axis=1)


# =====================================================================
# 4. PIPELINE: TF-IDF + ESTILOMETRÍA + PSICOLINGÜÍSTICA → SVM
# =====================================================================
procesador_caracteristicas = ColumnTransformer(transformers=[
    ('texto_tfidf', TfidfVectorizer(ngram_range=(1, 3), analyzer='word'), 'texto'),
    ('estilometria', 'passthrough', FEATURE_COLS)
])

pipeline_detector = Pipeline([
    ('preprocesamiento', procesador_caracteristicas),
    ('clasificador',     LinearSVC(class_weight='balanced', random_state=42))
])

pipeline_detector.fit(X_train_completo, todas_etiquetas)


# =====================================================================
# 5. FUNCIÓN DE ANÁLISIS DE UN MENSAJE
# =====================================================================
def analizar_mensaje(texto):
    """Evalúa un mensaje y muestra el resultado con desglose de métricas."""
    df_nuevo         = pd.DataFrame({'texto': [texto]})
    X_estilo         = calcular_metricas_estilo(df_nuevo['texto'])
    X_completo       = pd.concat([df_nuevo, X_estilo], axis=1)

    prediccion = pipeline_detector.predict(X_completo)[0]
    perfil     = "Menor de Edad (Perfil Joven)" if prediccion == 0 else "Adulto (Perfil Maduro)"

    imp   = X_estilo['psicoling_impulsividad'].values[0]
    anal  = X_estilo['psicoling_analitico'].values[0]
    ratio = X_estilo['psicoling_ratio_impulsivo'].values[0]
    caps  = X_estilo['caps_ratio'].values[0]
    reps  = X_estilo['caracteres_repetidos'].values[0]
    emjs  = X_estilo['emojis_y_signos'].values[0]
    longp = X_estilo['long_promedio_palabra'].values[0]

    icono = "🧒" if prediccion == 0 else "🧑"

    print("╔══════════════════════════════════════════════════╗")
    print(f"║  {icono}  RESULTADO: {perfil:<38}               ║")
    print(" ║─────────────────────────────────────────────────║")
    print(" ║  MÉTRICAS ESTILOMÉTRICAS                        ║")
    print(f"║    Ratio de mayúsculas     : {caps:.4f}         ║")
    print(f"║    Caracteres repetidos    : {reps:<4.0f}       ║")
    print(f"║    Ratio emojis/signos     : {emjs:.4f}         ║")
    print(f"║    Longitud prom. palabra  : {longp:.2f}        ║")
    print(" ║─────────────────────────────────────────────────║")
    print( "║  RASGOS PSICOLINGÜÍSTICOS                       ║")
    print(f"║    Impulsividad emocional  : {imp:<4.0f}        ║")
    print(f"║    Tono analítico/laboral  : {anal:<4.0f}       ║")
    print(f"║    Ratio impulsivo (0→1)   : {ratio:.2f}        ║")
    print("╚══════════════════════════════════════════════════╝")


# =====================================================================
# 6. MODO INTERACTIVO — el usuario escribe sus propios mensajes
# =====================================================================
print()
print("╔═══════════════════════════════════════════════════════╗")
print("║         DETECTOR DE PERFIL DE EDAD                    ║")
print("║   Modelo entrenado y listo · Escribe 'salir' para     ║")
print("║   terminar · Escribe 'demo' para ver ejemplos         ║")
print("╚═══════════════════════════════════════════════════════╝")

DEMO_MENSAJES = [
    "omg me hackearon la cuenta!! q injusto no se vale 😭🤬",
    "Estimados colegas, adjunto el informe financiero para su revisión.",
    "jaja q oso me cacharon copiando en el examen 😳",
    "oye perdona, estaba en una reunión con el cliente, ahorita te marco",
]

while True:
    print()
    try:
        entrada = input("✏️  Escribe un mensaje: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\n👋 ¡Hasta luego!")
        break

    if not entrada:
        print("⚠️  El mensaje está vacío. Escribe algo para analizar.")
        continue

    if entrada.lower() == "salir":
        print("\n👋 ¡Hasta luego!")
        break

    if entrada.lower() == "demo":
        print("\n📋 Ejecutando mensajes de demostración...\n")
        for msg in DEMO_MENSAJES:
            print(f'  → "{msg}"')
            analizar_mensaje(msg)
        continue

    analizar_mensaje(entrada)