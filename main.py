"""
Reconocimiento de gestos (MediaPipe) y de voz (Vosk) en paralelo.

Estructura de carpetas esperada:
    main.py
    models/
        gesture_recognizer.task
        vosk-model-small-es-...   (cualquier carpeta que empiece con "vosk-model")

Salir: tecla "q" en la ventana, o decir "salir".
"""

import json
import queue
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import sounddevice as sd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from vosk import KaldiRecognizer, Model, SetLogLevel

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
BASE = Path(__file__).parent
MODELOS = BASE / "models"
MODELO_GESTOS = MODELOS / "gesture_recognizer.task"


def buscar_modelo_vosk() -> Path:
    """Devuelve la primera carpeta dentro de models/ que empiece con 'vosk-model'."""
    for carpeta in MODELOS.iterdir():
        if carpeta.is_dir() and carpeta.name.startswith("vosk-model"):
            return carpeta
    raise FileNotFoundError("No encontré un modelo de Vosk dentro de la carpeta models/")


# --------------------------------------------------------------------------
# 1) ACCIONES: lo que tu programa sabe hacer
#    Cambia el contenido de estas funciones por lo que necesite tu proyecto.
# --------------------------------------------------------------------------
estado = {"mensaje": "Listo", "volumen": 5}
detener = threading.Event()


def accion_encender():
    estado["mensaje"] = "Luz encendida"


def accion_apagar():
    estado["mensaje"] = "Luz apagada"


def accion_subir_volumen():
    estado["volumen"] = min(10, estado["volumen"] + 1)
    estado["mensaje"] = f"Volumen: {estado['volumen']}"


def accion_bajar_volumen():
    estado["volumen"] = max(0, estado["volumen"] - 1)
    estado["mensaje"] = f"Volumen: {estado['volumen']}"


def accion_siguiente():
    estado["mensaje"] = "Siguiente"


def accion_anterior():
    estado["mensaje"] = "Anterior"


def accion_pausa():
    estado["mensaje"] = "Pausa"


def accion_continuar():
    estado["mensaje"] = "Continuar"


def accion_salir():
    detener.set()


ACCIONES = {
    "encender": accion_encender,
    "apagar": accion_apagar,
    "subir_volumen": accion_subir_volumen,
    "bajar_volumen": accion_bajar_volumen,
    "siguiente": accion_siguiente,
    "anterior": accion_anterior,
    "pausa": accion_pausa,
    "continuar": accion_continuar,
    "salir": accion_salir,
}

# --------------------------------------------------------------------------
# 2) MAPEOS: frase hablada -> acción, y gesto -> acción
#    Varias frases pueden llevar a la misma acción.
#    Usa palabras SIN acento y en minúsculas, como las escribe el modelo de Vosk.
# --------------------------------------------------------------------------
COMANDOS_VOZ = {
    "enciende la luz": "encender",
    "encender luz": "encender",
    "apaga la luz": "apagar",
    "apagar luz": "apagar",
    "sube el volumen": "subir_volumen",
    "baja el volumen": "bajar_volumen",
    "siguiente": "siguiente",
    "anterior": "anterior",
    "pausa": "pausa",
    "continuar": "continuar",
    "salir": "salir",
}

# Nombres de gestos que entrega MediaPipe: Closed_Fist, Open_Palm, Pointing_Up,
# Thumb_Down, Thumb_Up, Victory, ILoveYou.
COMANDOS_GESTO = {
    "Open_Palm": "pausa",
    "Closed_Fist": "continuar",
    "Thumb_Up": "subir_volumen",
    "Thumb_Down": "bajar_volumen",
    "Victory": "siguiente",
}

# --------------------------------------------------------------------------
# 3) VOZ: hilo que escucha el micrófono con Vosk
# --------------------------------------------------------------------------
eventos = queue.Queue()  # aquí llegan los eventos de voz y de gestos


def hilo_voz():
    SetLogLevel(-1)  # silencia los mensajes internos de Vosk
    modelo = Model(str(buscar_modelo_vosk()))

    # Gramática: Vosk solo reconocerá estas frases (más precisión).
    # "[unk]" permite descartar lo que no sea un comando.
    gramatica = json.dumps(list(COMANDOS_VOZ) + ["[unk]"], ensure_ascii=False)
    reconocedor = KaldiRecognizer(modelo, 16000, gramatica)

    audio = queue.Queue()

    def callback(datos, frames, tiempo, estado_audio):
        audio.put(bytes(datos))

    with sd.RawInputStream(
        samplerate=16000, blocksize=4000, dtype="int16", channels=1, callback=callback
    ):
        while not detener.is_set():
            try:
                datos = audio.get(timeout=0.2)
            except queue.Empty:
                continue
            if reconocedor.AcceptWaveform(datos):
                texto = json.loads(reconocedor.Result()).get("text", "").strip()
                if texto in COMANDOS_VOZ:
                    eventos.put(("voz", COMANDOS_VOZ[texto], texto))


# --------------------------------------------------------------------------
# 4) GESTOS: ciclo principal con la cámara (en el hilo principal)
# --------------------------------------------------------------------------
FRAMES_CONFIRMACION = 8   # cuántos cuadros seguidos debe verse el gesto
ENFRIAMIENTO = 1.5        # segundos mínimos entre dos disparos del mismo gesto
CONFIANZA_MINIMA = 0.6


def crear_reconocedor_gestos():
    opciones = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODELO_GESTOS)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    return vision.GestureRecognizer.create_from_options(opciones)


def dibujar(frame, gesto_actual, ultimo_evento, landmarks):
    alto, ancho = frame.shape[:2]
    if landmarks:
        for punto in landmarks:
            cv2.circle(frame, (int(punto.x * ancho), int(punto.y * alto)), 3, (0, 255, 0), -1)
    cv2.putText(frame, f"Gesto: {gesto_actual}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Estado: {estado['mensaje']}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Ultimo: {ultimo_evento}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)


def main():
    hilo = threading.Thread(target=hilo_voz, daemon=True)
    hilo.start()

    camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW acelera el inicio en Windows
    if not camara.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    inicio = time.monotonic()
    gesto_previo, contador = "None", 0
    ultimo_disparo = {}
    ultimo_evento = "-"

    with crear_reconocedor_gestos() as reconocedor:
        while not detener.is_set():
            ok, frame = camara.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # efecto espejo

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            marca_ms = int((time.monotonic() - inicio) * 1000)  # debe ir creciendo
            resultado = reconocedor.recognize_for_video(imagen, marca_ms)

            gesto_actual = "None"
            landmarks = None
            if resultado.gestures:
                categoria = resultado.gestures[0][0]
                if categoria.score >= CONFIANZA_MINIMA:
                    gesto_actual = categoria.category_name
                landmarks = resultado.hand_landmarks[0]

            # Confirmación: el gesto debe mantenerse varios cuadros seguidos
            if gesto_actual == gesto_previo:
                contador += 1
            else:
                gesto_previo, contador = gesto_actual, 1

            if contador == FRAMES_CONFIRMACION and gesto_actual in COMANDOS_GESTO:
                ahora = time.monotonic()
                if ahora - ultimo_disparo.get(gesto_actual, 0) > ENFRIAMIENTO:
                    ultimo_disparo[gesto_actual] = ahora
                    eventos.put(("gesto", COMANDOS_GESTO[gesto_actual], gesto_actual))

            # Procesar los eventos pendientes (de voz y de gestos)
            while not eventos.empty():
                origen, accion, detalle = eventos.get()
                ACCIONES[accion]()
                ultimo_evento = f"{origen}: {detalle}"
                print(f"[{origen}] {detalle} -> {accion}")

            dibujar(frame, gesto_actual, ultimo_evento, landmarks)
            cv2.imshow("Gestos y voz", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                detener.set()

    camara.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
