"""
Reconocimiento de gestos (MediaPipe) y de voz (Vosk) con una palabra clave.
en una interfaz a pantalla completa con botones

Estructura de carpetas esperada:
    main.py
    models/
        gesture_recognizer.task
        vosk-model-small-es-...   (cualquier carpeta que empiece con "vosk-model")

    Uso:
    - El microfono siempre escucha, pero solo espera la PALABRA_CLAVE.
    - Al oirla, el circulo cambia de color y tienes unos segundos para decir un comando.
    - El boton de la flecha (abajo, centro) abre/cierra el panel de botones.
    - Salir: tecla "q" o ESC, el boton Salir, o decir la palabra clave y luego "salir".

    Salir: tecla "q" en la ventana, o decir "salir".
"""

import ctypes
import json
import math
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

PALABRA_CLAVE = "asistente"
SEGUNDOS_ESCUCHA = 7

FRAMES_CONFIRMACION = 8       # cuadros seguidos para confirmar un gesto
ENFRIAMIENTO = 1.5            # segundos minimos entre dos disparos del mismo gesto
CONFIANZA_MINIMA = 0.6

VENTANA = "Gestos y voz"

# Colores en formato rgb
VERDE = (60,170,80)
NARANJA = (0,140, 255)
GRIS_CLARO = (205, 205, 205)
BLANCO = (255, 255, 255)
NEGRO = (0,0,0)


def buscar_modelo_vosk() -> Path:
    """Devuelve la primera carpeta dentro de models/ que empiece con 'vosk-model'."""
    for carpeta in MODELOS.iterdir():
        if carpeta.is_dir() and carpeta.name.startswith("vosk-model"):
            return carpeta
    raise FileNotFoundError("No encontré un modelo de Vosk dentro de la carpeta models/")

def tamano_pantalla():
    try:
        usuario = ctypes.winddl.user32
        usuario.SetProcessDPIAware()
        return usuario.GetSystemMetrics(0), usuario.GetSystemMetrics(1)
    except Exception:
        return 1200, 720

# --------------------------------------------------------------------------
# 1) ACCIONES: lo que tu programa sabe hacer
#    Cambia el contenido de estas funciones por lo que necesite tu proyecto.
# --------------------------------------------------------------------------
estado = {
    "mensaje": "Listo", 
    "volumen": 5,
    "voz" : "espera",
    "activo_hasta": 0.0,
}
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

# Botones del panel: (texto, accion). OpenCV no dibuja acentos ni enes: evitalos.
BOTONES_PANEL = [
    ("Luz ON", "encender"),
    ("Luz OFF", "apagar"),
    ("Vol +", "subir_volumen"),
    ("Vol -", "bajar_volumen"),
    ("Anterior", "anterior"),
    ("Siguiente", "siguiente"),
    ("Pausa", "pausa"),
    ("Salir", "salir"),
]

# --------------------------------------------------------------------------
# 3) VOZ: hilo que escucha el micrófono con Vosk. El microfono escucha pero solo reacciona a la palabra clave
# --------------------------------------------------------------------------
eventos = queue.Queue()  # aquí llegan los eventos de voz y de gestos


def hilo_voz():
    SetLogLevel(-1)  # silencia los mensajes internos de Vosk
    modelo = Model(str(buscar_modelo_vosk()))

    #Dos reconocedores con gramaticas distintas
    # - rec_clave: solo conoce la palabra clave (modo espera, siempre activo)
    # - rec_cmd:  solo conoce los comandos (modo activo, unos segundos)
    #"[unk]" permite descartar cualquier otra cosa que se diga

    rec_clave = KaldiRecognizer(
        modelo, 16000, json.dumps([PALABRA_CLAVE, "[unk]"], ensure_ascii=False)
    )
    rec_cmd = KaldiRecognizer(
        modelo, 16000, json.dumps(list(COMANDOS_VOZ) + ["[unk]"], ensure_ascii=False)
    )

    audio = queue.Queue()

    def callback(datos, frames, tiempo, estado_audio):
        audio.put(bytes(datos))

    activo_hasta = 0.0

    with sd.RawInputStream(
        samplerate=16000, blocksize=4000, dtype="int16", channels=1, callback=callback
    ):
        while not detener.is_set():
            try:
                datos = audio.get(timeout=0.2)
            except queue.Empty:
                continue
            ahora = time.monotonic()

            if ahora < activo_hasta:
                texto = json.loads(rec_cmd.Result()).get("text", "").strip()
                if texto in COMANDOS_VOZ:
                    eventos.put(("voz", COMANDOS_VOZ[texto], texto))
                    activo_hasta = 0.0
                    estado["voz"] = "espera"
                    rec_clave.Reset()
            else:
                # MODO ESPERA: solo la palabra clave
                if estado["voz"] == "activo":  # se acabo el tiempo sin comando
                    estado["voz"] = "espera"
                    rec_clave.Reset()

                if rec_clave.AcceptWaveform(datos):
                    texto = json.loads(rec_clave.Result()).get("text", "")
                else:
                    # El resultado parcial permite reaccionar sin esperar el silencio
                    texto = json.loads(rec_clave.PartialResult()).get("partial", "")

                if PALABRA_CLAVE in texto.split():
                    rec_clave.Reset()
                    rec_cmd.Reset()
                    activo_hasta = ahora + SEGUNDOS_ESCUCHA
                    estado["activo_hasta"] = activo_hasta
                    estado["voz"] = "activo"


# --------------------------------------------------------------------------
# 4) INTERFAZ: pantalla completa, indicador de microfono y panel de botones
# --------------------------------------------------------------------------
ui = {"panel": False, "botones": []}  # "botones" se recalcula en cada cuadro


def al_hacer_clic(evento, x, y, flags, param):
    if evento != cv2.EVENT_LBUTTONDOWN:
        return
    for x1, y1, x2, y2, accion in ui["botones"]:
        if x1 <= x <= x2 and y1 <= y <= y2:
            if accion == "__panel__":
                ui["panel"] = not ui["panel"]
            else:
                eventos.put(("boton", accion, accion))
            return


def ajustar_a_pantalla(frame, ancho, alto):
    """Escala la imagen para cubrir toda la pantalla, recortando lo que sobre."""
    h, w = frame.shape[:2]
    escala = max(ancho / w, alto / h)
    nuevo_w, nuevo_h = math.ceil(w * escala), math.ceil(h * escala)
    redimensionada = cv2.resize(frame, (nuevo_w, nuevo_h))
    x0, y0 = (nuevo_w - ancho) // 2, (nuevo_h - alto) // 2
    return redimensionada[y0:y0 + alto, x0:x0 + ancho]


def rect_translucido(lienzo, x1, y1, x2, y2, color, alfa):
    capa = lienzo.copy()
    cv2.rectangle(capa, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(capa, alfa, lienzo, 1 - alfa, 0, lienzo)


def texto_centrado(lienzo, texto, cx, cy, escala, color, grosor=2):
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
    cv2.putText(lienzo, texto, (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def dibujar_interfaz(lienzo, gesto_actual, ultimo_evento):
    alto, ancho = lienzo.shape[:2]
    k = alto / 720  # factor de escala segun la resolucion
    ui["botones"] = []

    # --- Informacion arriba a la izquierda ---
    rect_translucido(lienzo, 0, 0, int(520 * k), int(120 * k), NEGRO, 0.45)
    for i, linea in enumerate([
        f"Gesto: {gesto_actual}",
        f"Estado: {estado['mensaje']}",
        f"Ultimo: {ultimo_evento}",
    ]):
        cv2.putText(lienzo, linea, (int(15 * k), int((35 + 35 * i) * k)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * k, BLANCO, 2, cv2.LINE_AA)

    # --- Franja inferior ---
    rect_translucido(lienzo, 0, alto - int(190 * k), ancho, alto, NEGRO, 0.35)

    # Indicador de microfono (circulo)
    cx, cy, radio = ancho // 2, alto - int(115 * k), int(38 * k)
    activo = estado["voz"] == "activo"
    cv2.circle(lienzo, (cx, cy), radio, NARANJA if activo else VERDE, -1, cv2.LINE_AA)
    cv2.circle(lienzo, (cx, cy), radio + 3, GRIS_CLARO, max(2, int(4 * k)), cv2.LINE_AA)

    if activo:
        restante = max(0.0, estado["activo_hasta"] - time.monotonic())
        etiqueta = f"Escuchando comando... {restante:.0f}s"
    else:
        etiqueta = f'Di "{PALABRA_CLAVE}"'
    texto_centrado(lienzo, etiqueta, cx, cy + radio + int(22 * k), 0.6 * k, BLANCO)

    # Boton de flecha (abre/cierra el panel)
    bx1, by1 = cx - int(26 * k), alto - int(32 * k)
    bx2, by2 = cx + int(26 * k), alto - int(10 * k)
    cv2.rectangle(lienzo, (bx1, by1), (bx2, by2), GRIS_CLARO, -1)
    mx = cx
    arriba, abajo = (mx, by1 + int(4 * k)), (mx, by2 - int(4 * k))
    if ui["panel"]:
        cv2.arrowedLine(lienzo, arriba, abajo, NEGRO, 2, tipLength=0.5)
    else:
        cv2.arrowedLine(lienzo, abajo, arriba, NEGRO, 2, tipLength=0.5)
    ui["botones"].append((bx1, by1, bx2, by2, "__panel__"))

    # --- Panel de botones ---
    if ui["panel"]:
        n = len(BOTONES_PANEL)
        hueco = int(10 * k)
        bw = min(int(130 * k), (ancho - 40 - hueco * (n - 1)) // n)
        bh = int(52 * k)
        total = n * bw + (n - 1) * hueco
        x = (ancho - total) // 2
        y1 = cy - radio - bh - int(25 * k)

        rect_translucido(lienzo, x - hueco, y1 - hueco, x + total + hueco,
                         y1 + bh + hueco, NEGRO, 0.55)
        for texto, accion in BOTONES_PANEL:
            cv2.rectangle(lienzo, (x, y1), (x + bw, y1 + bh), GRIS_CLARO, -1)
            texto_centrado(lienzo, texto, x + bw // 2, y1 + bh // 2, 0.6 * k, NEGRO)
            ui["botones"].append((x, y1, x + bw, y1 + bh, accion))
            x += bw + hueco


# --------------------------------------------------------------------------
# 5) GESTOS + CICLO PRINCIPAL (hilo principal)
# --------------------------------------------------------------------------
def crear_reconocedor_gestos():
    opciones = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODELO_GESTOS)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    return vision.GestureRecognizer.create_from_options(opciones)


def main():
    ancho_p, alto_p = tamano_pantalla()

    threading.Thread(target=hilo_voz, daemon=True).start()

    camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW acelera el inicio en Windows
    if not camara.isOpened():
        raise RuntimeError("No se pudo abrir la camara")
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(VENTANA, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(VENTANA, al_hacer_clic)

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
            if resultado.gestures:
                categoria = resultado.gestures[0][0]
                if categoria.score >= CONFIANZA_MINIMA:
                    gesto_actual = categoria.category_name
                # Puntos de la mano, dibujados sobre la imagen original
                alto_f, ancho_f = frame.shape[:2]
                for punto in resultado.hand_landmarks[0]:
                    cv2.circle(frame, (int(punto.x * ancho_f), int(punto.y * alto_f)),
                               4, (0, 255, 0), -1)

            # Confirmacion: el gesto debe mantenerse varios cuadros seguidos
            if gesto_actual == gesto_previo:
                contador += 1
            else:
                gesto_previo, contador = gesto_actual, 1

            if contador == FRAMES_CONFIRMACION and gesto_actual in COMANDOS_GESTO:
                ahora = time.monotonic()
                if ahora - ultimo_disparo.get(gesto_actual, 0) > ENFRIAMIENTO:
                    ultimo_disparo[gesto_actual] = ahora
                    eventos.put(("gesto", COMANDOS_GESTO[gesto_actual], gesto_actual))

            # Procesar eventos pendientes (voz, gestos y botones)
            while not eventos.empty():
                origen, accion, detalle = eventos.get()
                ACCIONES[accion]()
                ultimo_evento = f"{origen}: {detalle}"
                print(f"[{origen}] {detalle} -> {accion}")

            # Imagen a pantalla completa + interfaz encima
            lienzo = ajustar_a_pantalla(frame, ancho_p, alto_p)
            dibujar_interfaz(lienzo, gesto_actual, ultimo_evento)
            cv2.imshow(VENTANA, lienzo)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):  # q o ESC
                detener.set()

    camara.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()