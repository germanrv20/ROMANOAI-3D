"""
ROMANOAI - Aplicación de Realidad Aumentada Educativa

Este programa combina reconocimiento facial, realidad aumentada con marcadores ARUCO
y comandos de voz para crear una experiencia educativa sobre estructuras históricas romanas.
"""

import cv2
import os
import pathlib
import numpy as np
import cuia  # Módulo personalizado para manejo de modelos 3D y realidad aumentada
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageEnhance
import speech_recognition as sr
import pyttsx3
import sqlite3
import json
import sys
from gtts import gTTS
import tempfile
import pygame
import time

# ===========================================
# CONFIGURACIÓN INICIAL Y CONSTANTES
# ===========================================

# Directorio base de la aplicación
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resource_path(relative_path):
    """Obtiene la ruta correcta de los archivos, tanto en desarrollo como en .exe"""
    if getattr(sys, 'frozen', False):  # Si está ejecutándose como .exe
        base_path = sys._MEIPASS  # Carpeta temporal de PyInstaller
    else:  # Si está en desarrollo
        base_path = os.path.abspath(".")  # Usa la ruta normal
    
    return os.path.join(base_path, relative_path)

# Cargar información de los modelos 3D desde archivo JSON
jsonfolder = resource_path(os.path.join("dist", "datosModelos.json"))
with open(jsonfolder, "r", encoding="utf-8") as f:
    info_modelos = json.load(f)

# ===========================================
# CARGAR MODELOS DE VISIÓN POR COMPUTADOR
# ===========================================

# Cargar modelo de detección de caras (Caffe)
red = cv2.dnn.readNetFromCaffe("dnn/deploy.prototxt", "dnn/res10_300x300_ssd_iter_140000.caffemodel")

# Cargar modelo de reconocimiento facial (OpenCV)
fr = cv2.FaceRecognizerSF.create("dnn/face_recognition_sface_2021dec.onnx", "")

# Cargar detector de caras YuNet
detector_caras = cv2.FaceDetectorYN.create("dnn/face_detection_yunet_2023mar.onnx", 
                                         config="", 
                                         input_size=(320, 320), 
                                         score_threshold=0.7)

# Configuración inicial de dispositivos
myCam = 1  # Índice de cámara por defecto
mymicro = 1  # Índice de micrófono por defecto

# Variables de estado global
ventana_inicial_cerrada = False
idModelo = 0  # ID del modelo 3D actualmente visualizado
reconocido = False  # Indica si se ha reconocido un usuario
elimina = False  # Bandera para eliminar usuario

# Configuración del motor de voz
engine_pyttsx3 = None
engine_lock = threading.Lock()  # Lock para seguridad en hilos

# ===========================================
# BASE DE DATOS DE USUARIOS
# ===========================================

# Diccionario en memoria con información de usuarios
usuarios = {
    "Zakaria": {"pais": "Marruecos", "idioma": "ingles", "vector": None},
    "PabloGolfo": {"pais": "España.", "idioma": "Italiano", "vector": None},
}

# ===========================================
# FUNCIONES DE MANEJO DE USUARIOS
# ===========================================

def obtener_info_usuario(nombre):
    """
    Obtiene la información de un usuario desde la base de datos SQLite.
    
    Args:
        nombre (str): Nombre del usuario a buscar
    
    Returns:
        dict: Diccionario con información del usuario o None si no existe
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    basefolder = os.path.join(BASE_DIR, "dist","usuarios.db")
    conn = sqlite3.connect(basefolder)
    cursor = conn.cursor()

    cursor.execute("SELECT pais, idioma FROM usuarios WHERE nombre = ?", (nombre,))
    resultado = cursor.fetchone()
    conn.close()

    if resultado:
        pais, idioma = resultado
        return {"pais": pais, "idioma": idioma}
    else:
        return None

def borrar_usuario(nombre):
    """
    Elimina un usuario de la base de datos y del sistema de archivos.
    
    Args:
        nombre (str): Nombre del usuario a eliminar
    """
    # Ruta a la base de datos
    ruta_db = os.path.join(BASE_DIR, "dist","usuarios.db")
    conn = sqlite3.connect(ruta_db)
    cursor = conn.cursor()

    # Eliminar de la base de datos
    cursor.execute("DELETE FROM usuarios WHERE nombre = ?", (nombre,))
    conn.commit()
    filas_afectadas = cursor.rowcount
    conn.close()

    # Eliminar del diccionario en memoria si existe
    if nombre in usuarios:
        del usuarios[nombre]

    # Eliminar imagen del usuario
    ruta_imagen = os.path.join(BASE_DIR, "dist","media", "Usuarios", f"{nombre}.jpg")
    if os.path.exists(ruta_imagen):
        os.remove(ruta_imagen)
        print(f"🗑️ Imagen de {nombre} eliminada.")

    # Reporte final
    if filas_afectadas > 0:
        print(f"✅ Usuario '{nombre}' borrado correctamente de la base de datos.")
    else:
        print(f"⚠️ No se encontró ningún usuario con el nombre '{nombre}'.")

def cargar_usuarios():
    """
    Carga los usuarios registrados desde el directorio de imágenes al sistema.
    Genera los vectores faciales para cada usuario encontrado.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    usuarios_folder = os.path.join(BASE_DIR, "dist", "media", "Usuarios")
    if not os.path.exists(usuarios_folder):
        os.makedirs(usuarios_folder)
    
    lista = os.listdir(usuarios_folder)
    for f in lista:
        ruta = os.path.join(usuarios_folder, f)
       
        if not os.path.isfile(ruta):
            print(f"❌ El archivo no existe: {ruta}")
            continue
        
        nombre = pathlib.Path(f).stem    
        img = cv2.imread(ruta)
        if img is None:
            print(f"⚠️ No se pudo cargar la imagen {ruta}")
            continue

        h, w, _ = img.shape
        detector_caras.setInputSize((w, h))
        ret, caranueva = detector_caras.detect(img)
        if ret and caranueva is not None:
            caracrop = fr.alignCrop(img, caranueva[0])
            codcara = fr.feature(caracrop)
            if nombre not in usuarios:
                usuarios[nombre] = {"pais": "Desconocido", "idioma": "Desconocido", "vector": None}
            usuarios[nombre]["vector"] = codcara
            usuarios[nombre]["imagen"] = img.copy()  
            print(f"✅ Cargado vector de {nombre}")
        else:
            print(f"❌ No se detectó cara en {nombre}")

# Cargar usuarios al iniciar la aplicación
cargar_usuarios()

# ===========================================
# CONFIGURACIÓN DE MODELOS 3D Y ARUCO
# ===========================================

# Mapeo de IDs de marcadores ARUCO a modelos 3D
modelos_por_id = { 
    0: "templo_de_saturno_roma.glb", 
    1: "foro-romano.glb", 
    2: "libarna_-_anfiteatro.glb", 
    3: "libarna_-_foro.glb", 
    4: "ciudadFortificada.glb", 
    5: "casasRomanas.glb",
}

# Configuración de detección de marcadores ARUCO
diccionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
detector_aruco = cv2.aruco.ArucoDetector(diccionario)

# ===========================================
# FUNCIONES AUXILIARES DE VISIÓN POR COMPUTADOR
# ===========================================

def origen(TAM):
    """
    Genera puntos 3D para un marcador ARUCO de tamaño especificado.
    
    Args:
        TAM (float): Tamaño del marcador en metros
    
    Returns:
        np.array: Array con los 4 puntos del marcador en 3D
    """
    return np.array([[-TAM/2.0, -TAM/2.0, 0.0],
                     [-TAM/2.0,  TAM/2.0, 0.0],
                     [ TAM/2.0,  TAM/2.0, 0.0],
                     [ TAM/2.0, -TAM/2.0, 0.0]])

def proyeccion(puntos, rvec, tvec, cameraMatrix, distCoeffs):
    """
    Proyecta puntos 3D a 2D usando los parámetros de la cámara.
    
    Args:
        puntos (np.array): Puntos 3D a proyectar
        rvec (np.array): Vector de rotación
        tvec (np.array): Vector de traslación
        cameraMatrix (np.array): Matriz de cámara
        distCoeffs (np.array): Coeficientes de distorsión
    
    Returns:
        list: Lista de puntos 2D proyectados
    """
    puntos = np.array(puntos, dtype=np.float32)
    proyectados, _ = cv2.projectPoints(puntos, rvec, tvec, cameraMatrix, distCoeffs)
    return [tuple(map(int, p[0])) for p in proyectados]

def fov(cameraMatrix, ancho, alto):
    """
    Calcula el campo de visión (FOV) de la cámara.
    
    Args:
        cameraMatrix (np.array): Matriz de cámara
        ancho (int): Ancho del frame
        alto (int): Alto del frame
    
    Returns:
        float: Ángulo de campo de visión en grados
    """
    if ancho > alto:
        f = cameraMatrix[1, 1]
        fov_rad = 2 * np.arctan(alto / (2 * f))
    else:
        f = cameraMatrix[0, 0]
        fov_rad = 2 * np.arctan(ancho / (2 * f))
    return np.rad2deg(fov_rad)

def fromOpencvToPygfx(rvec, tvec):
    """
    Convierte la pose de OpenCV al formato de PyGFX.
    
    Args:
        rvec (np.array): Vector de rotación de OpenCV
        tvec (np.array): Vector de traslación de OpenCV
    
    Returns:
        np.array: Matriz de transformación 4x4 para PyGFX
    """
    pose = np.eye(4)
    pose[0:3, 3] = tvec.T
    pose[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
    pose[1:3] *= -1
    pose = np.linalg.inv(pose)
    return pose

def detectarPose(frame, tam):
    """
    Detecta marcadores ARUCO en un frame y calcula su pose.
    
    Args:
        frame (np.array): Imagen donde buscar marcadores
        tam (float): Tamaño real del marcador en metros
    
    Returns:
        tuple: (bool, dict) Indicador de éxito y diccionario con poses por ID
    """
    bboxs, ids, _ = detector_aruco.detectMarkers(frame)
    if ids is not None:
        objPoints = np.array([[-tam/2.0, tam/2.0, 0.0],
                              [tam/2.0, tam/2.0, 0.0],
                              [tam/2.0, -tam/2.0, 0.0],
                              [-tam/2.0, -tam/2.0, 0.0]])
        resultado = {}
        for i in range(len(ids)):
            ret, rvec, tvec = cv2.solvePnP(objPoints, bboxs[i], cameraMatrix, distCoeffs)
            if ret:
                resultado[ids[i][0]] = (rvec, tvec)
        return (True, resultado)
    return (False, None)

def realidadMixta(frame, ancho, alto):
    """
    Combina la realidad aumentada con el frame de la cámara.
    
    Args:
        frame (np.array): Frame de la cámara
        ancho (int): Ancho del frame
        alto (int): Alto del frame
    
    Returns:
        np.array: Frame con los modelos 3D renderizados
    """
    global idModelo
    ret, poses = detectarPose(frame, 0.19)
    resultado = frame

    if ret and poses:
        imagen_render = np.zeros((alto, ancho, 4), dtype=np.uint8)

        for id_detectado, (rvec, tvec) in poses.items():
            if id_detectado in modelos and id_detectado in escenas:
                idModelo = id_detectado
                modelo = modelos[id_detectado]
                escena = escenas[id_detectado]

                M = fromOpencvToPygfx(rvec, tvec)
                escena.actualizar_camara(M)

                imagen_modelo = escena.render()
                imagen_modelo_bgr = cv2.cvtColor(imagen_modelo, cv2.COLOR_RGBA2BGRA)

                imagen_render = cuia.alphaBlending(imagen_modelo_bgr, imagen_render)

        resultado = cuia.alphaBlending(imagen_render, frame)

    return resultado

# ===========================================
# CONFIGURACIÓN DE VOZ Y AUDIO
# ===========================================

# Mapeo de idiomas a códigos normalizados
idioma_normalizado = {
    "español": "es",
    "español méxico": "es-mx",
    "inglés": "en",
    "italiano": "it"
}

# Mapeo de idiomas a voces en pyttsx3
voices_map = {
    "es": 1,  # Helena - español
    "en": 2,  # Zira - inglés
    "it": 0,  # Elsa - italiano
}

def reproducir_audio(ruta):
    """
    Reproduce un archivo de audio usando pygame.
    
    Args:
        ruta (str): Ruta al archivo de audio
    """
    pygame.mixer.init()
    pygame.mixer.music.load(ruta)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

def hablar(texto, idioma="es", velocidad=120, motor="pyttsx3"):
    """
    Sintetiza voz a partir de texto, con soporte para múltiples idiomas.
    Usa gTTS para español (con fallback a pyttsx3) y pyttsx3 para otros idiomas.
    
    Args:
        texto (str): Texto a convertir en voz
        idioma (str): Idioma de salida ('es', 'en', 'it')
        velocidad (int): Velocidad del habla (solo pyttsx3)
        motor (str): Motor a usar ('gtts' o 'pyttsx3')
    """
    if idioma == "es":
        try:
            tts = gTTS(text=texto, lang=idioma)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                temp_path = f.name
            tts.save(temp_path)
            time.sleep(0.1)
            reproducir_audio(temp_path)
        except Exception as e:
            print(f"[gTTS] Error: {e}. Reintentando con pyttsx3.")
            hablar(texto, idioma, velocidad, motor="pyttsx3")
    elif(motor == "pyttsx3" or idioma != "es" ):
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        idx = voices_map.get(idioma, 1)  # Si no encuentra idioma, usa español por defecto
        if idx < len(voices):
            engine.setProperty('voice', voices[idx].id)
        else:
            print(f"No se encontró voz para idioma '{idioma}', usando la voz por defecto.")
        
        engine.setProperty('rate', velocidad)
        engine.say(texto)
        engine.runAndWait()

# ===========================================
# RECONOCIMIENTO DE VOZ
# ===========================================

def escuchar_comandos():
    """
    Escucha continuamente comandos de voz usando el micrófono seleccionado.
    Los comandos reconocidos se envían a procesar_comando().
    """
    recognizer = sr.Recognizer()
    mic_index = mymicro
    mic = sr.Microphone(device_index=mic_index)
    print("Escuchando...")

    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, phrase_time_limit=4)
            try:
                idioma_google = "es-ES"  # valor por defecto
                if reconocido:
                    datos_user = usuarios.get(nombre)
                    idiomaU = datos_user["idioma"].lower()
                    if idiomaU == "español":
                        idioma_google = "es-ES"
                        print("TIPO IDIOMA: Español")
                    elif idiomaU == "ingles":
                        idioma_google = "en-US"
                        print("TIPO IDIOMA: Inglés")
                    elif idiomaU == "italiano":
                        idioma_google = "it-IT"
                        print("TIPO IDIOMA: Italiano")

                comando = recognizer.recognize_google(audio, language=idioma_google)
                print("Comando reconocido: ", comando)
                procesar_comando(comando)
            except sr.UnknownValueError:
                print("No se entendió el comando")
            except sr.RequestError as e:
                print(f"Error al conectarse al servicio de reconocimiento: {e}")

def procesar_comando(comando):
    """
    Procesa un comando de voz reconocido y ejecuta la acción correspondiente.
    
    Args:
        comando (str): Comando de voz reconocido
    """
    global elimina
    global nombre, reconocido
    comando = comando.lower()
    print(f"reconocido = {reconocido}")
    if(reconocido == True):
        datosUSer = usuarios.get(nombre)
        
        idiomaUSER = "es"

        idiomaU = datosUSer["idioma"].lower()
        print(f"Idioma : {idiomaU}")
        print(f"nombre  : {nombre}")

        if(idiomaU == "español"):
                idiomaUSER = "es" 
        elif(idiomaU == "ingles"):
                idiomaUSER = "en"
        elif(idiomaU == "italiano"):
                idiomaUSER = "it"
        
        if any(palabra in comando for palabra in ["explica", "explains", "spiega"]):
            try:
                texto = info_modelos[str(idModelo)]["explica"][idiomaUSER]
                hablar(texto, idioma=idiomaUSER, velocidad=150 if idiomaUSER == "es" else 150)
            except KeyError:
                hablar("No tengo información sobre el uso de este modelo en este idioma.", idioma=idiomaUSER)

        if any(palabra in comando for palabra in ["año", "year", "anno"]):
            try:
                texto = info_modelos[str(idModelo)]["año"][idiomaUSER]
                hablar(texto, idioma=idiomaUSER, velocidad=150 if idiomaUSER == "es" else 150)
            except KeyError:
                hablar("No tengo información sobre el uso de este modelo en este idioma.", idioma=idiomaUSER)
        if any(palabra in comando for palabra in ["uso", "use", "utilizzo"]):
            try:
                texto = info_modelos[str(idModelo)]["uso"][idiomaUSER]
                hablar(texto, idioma=idiomaUSER, velocidad=150 if idiomaUSER == "es" else 150)
            except KeyError:
                hablar("No tengo información sobre el uso de este modelo en este idioma.", idioma=idiomaUSER)
        if any(palabra in comando for palabra in ["contexto", "context", "contesto"]):
            try:
                texto = info_modelos[str(idModelo)]["contexto"][idiomaUSER]
                hablar(texto, idioma=idiomaUSER, velocidad=150 if idiomaUSER == "es" else 150)
            except KeyError:
                hablar("No tengo información sobre el uso de este modelo en este idioma.", idioma=idiomaUSER)
        if any(palabra in comando for palabra in ["curiosidad", "curiosity", "curiosità"]):
            try:
                texto = info_modelos[str(idModelo)]["curiosidad"][idiomaUSER]
                hablar(texto, idioma=idiomaUSER, velocidad=150 if idiomaUSER == "es" else 150)
            except KeyError:
                hablar("No tengo información sobre el uso de este modelo en este idioma.", idioma=idiomaUSER)
        if any(palabra in comando for palabra in ["estructura", "structure", "struttura"]):
            try:
                texto = info_modelos[str(idModelo)]["estructura"][idiomaUSER]
                hablar(texto, idioma=idiomaUSER, velocidad=150 if idiomaUSER == "es" else 150)
            except KeyError:
                hablar("No tengo información sobre el uso de este modelo en este idioma.", idioma=idiomaUSER)
        if any(palabra in comando for palabra in ["funcionamiento", "operation", "operazione"]):
            try:
                texto = info_modelos[str(idModelo)]["funcionamiento"][idiomaUSER]
                hablar(texto, idioma=idiomaUSER, velocidad=150 if idiomaUSER == "es" else 150)
            except KeyError:
                hablar("No tengo información sobre el uso de este modelo en este idioma.", idioma=idiomaUSER)
        if any(palabra in comando for palabra in ["mitología", "mythology", "mitologia"]):
            try:
                texto = info_modelos[str(idModelo)]["mitología"][idiomaUSER]
                hablar(texto, idioma=idiomaUSER, velocidad=150 if idiomaUSER == "es" else 150)
            except KeyError:
                hablar("No tengo información sobre el uso de este modelo en este idioma.", idioma=idiomaUSER)
        if any(palabra in comando for palabra in ["elimina usuario", "remove user", "rimove usuario"]):
            elimina = True
            elimina = True

# Iniciar el hilo para escuchar comandos de voz
thread = threading.Thread(target=escuchar_comandos)
thread.daemon = True
thread.start()

# ===========================================
# INTERFAZ GRÁFICA
# ===========================================

class SelectorDispositivos:
    """
    Ventana para seleccionar dispositivos de cámara y micrófono.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Selección de Dispositivos")
        self.root.geometry("400x300")
        
        # Variables
        self.camara_var = tk.StringVar(value="0")
        self.microfono_var = tk.StringVar(value="0")
        
        # Obtener lista de dispositivos
        self.camaras = self.obtener_dispositivos_video()
        self.microfonos = self.obtener_dispositivos_audio()
        
        # Crear interfaz
        self.crear_interfaz()
    
    def obtener_dispositivos_video(self):
        """Devuelve lista de cámaras disponibles"""
        dispositivos = []
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                dispositivos.append(f"Cámara {index}")
            cap.release()
            index += 1
        return dispositivos
    
    def obtener_dispositivos_audio(self):
        """Devuelve lista de micrófonos disponibles"""
        return [f"Micrófono {i}" for i in range(len(sr.Microphone.list_microphone_names()))]
    
    def crear_interfaz(self):
        """Crea los elementos de la interfaz de selección de dispositivos"""
        frame = tk.Frame(self.root, padx=20, pady=20)
        frame.pack(expand=True, fill="both")
        
        # Selección de cámara
        tk.Label(frame, text="Selecciona la cámara:").grid(row=0, column=0, sticky="w", pady=5)
        cam_menu = ttk.Combobox(frame, textvariable=self.camara_var, values=self.camaras)
        cam_menu.grid(row=0, column=1, pady=5, padx=10)
        
        # Selección de micrófono
        tk.Label(frame, text="Selecciona el micrófono:").grid(row=1, column=0, sticky="w", pady=5)
        mic_menu = ttk.Combobox(frame, textvariable=self.microfono_var, values=self.microfonos)
        mic_menu.grid(row=1, column=1, pady=5, padx=10)
        
        # Botón de confirmación
        btn_confirmar = tk.Button(frame, text="Confirmar", command=self.confirmar_seleccion,
                                width=15, height=2, bg="#4CAF50", fg="white")
        btn_confirmar.grid(row=2, column=0, columnspan=2, pady=20)
    
    def confirmar_seleccion(self):
        """Guarda la selección de dispositivos y abre la interfaz principal"""
        global myCam, mymicro
        
        # Extraer el número del dispositivo seleccionado
        myCam = int(self.camara_var.get().split()[-1])
        mymicro = int(self.microfono_var.get().split()[-1])
        
        # Start the voice command thread after devices are selected
        thread = threading.Thread(target=escuchar_comandos)
        thread.daemon = True
        thread.start()
        
        self.root.destroy()
        
        # Iniciar la aplicación principal con los dispositivos seleccionados
        root_main = tk.Tk()
        app = InterfazUsuario(root_main)
        root_main.mainloop()


# ===========================================
# INTERFAZ DE REGISTRO Y LOGIN
# ===========================================
class InterfazUsuario:
    """
    Clase principal que maneja la interfaz gráfica de la aplicación.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("ROMANOAI")
        self.root.geometry("800x600")
        
        # Inicializar dispositivos
        self.cap = None
        self.init_camera()  # Inicializar cámara
        
        if self.cap is None:
            self.root.quit()
            return

        # Configuración de cámara y modelos 3D
        self.configurar_camara_y_modelos()
        
        # Resto de la inicialización...
        self.crear_menu()
        self.mostrar_pantalla_inicio()
        
        # Variables para registro
        self.nombre_var = tk.StringVar()
        self.pais_var = tk.StringVar()
        self.idioma_var = tk.StringVar()
        
        # Configurar cierre limpio
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar_aplicacion)


    def eliminar_usuario_y_cerrar_app(self, nombre_usuario):
        borrar_usuario(nombre_usuario)  # Eliminar usuario
        # Liberar recursos y cerrar todo
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()
        sys.exit()


    def init_camera(self):
        """Método para inicializar la cámara seleccionada"""
        global myCam
        self.cap = cv2.VideoCapture(myCam, cuia.bestBackend(myCam))
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(myCam, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "No se pudo abrir la cámara")
                return None
        
        # Configurar resolución
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return self.cap
    def cargar_modelos_3d(self):
        self.modelos = {}
        self.escenas = {}
        
        for id_aruco, ruta in modelos_por_id.items():
            m = cuia.modeloGLTF(ruta)
            m.rotar((np.pi/2.0, 0, 0))
            m.escalar(0.3)
            m.flotar()
           
            anims = m.animaciones()
            if len(anims) > 0:
                m.animar(anims[0])

            escena = cuia.escenaPYGFX(fov(self.cameraMatrix, self.ancho, self.alto), self.ancho, self.alto)
            escena.agregar_modelo(m)
            escena.ilumina_modelo(m)
            escena.iluminar()

            self.modelos[id_aruco] = m
            self.escenas[id_aruco] = escena
        
    def crear_menu(self):
        menubar = tk.Menu(self.root)
        
        # Menú Usuario
        usuario_menu = tk.Menu(menubar, tearoff=0)
        usuario_menu.add_command(label="Registrarse", command=self.mostrar_pantalla_registro)
        usuario_menu.add_command(label="Iniciar Sesión", command=self.mostrar_pantalla_login)
        usuario_menu.add_separator()
        usuario_menu.add_command(label="Salir", command=self.root.quit)
        menubar.add_cascade(label="Usuario", menu=usuario_menu)
        
        self.root.config(menu=menubar)
    
    def mostrar_pantalla_inicio(self):
        self.limpiar_pantalla()
        
        frame = tk.Frame(self.root, bg="#f0f0f0", padx=20, pady=20)
        frame.pack(expand=True, fill="both")
        
        label = tk.Label(frame, text="ROMANOAI", 
                        font=("Arial", 18), bg="#f0f0f0")
        label.pack(pady=20)
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(BASE_DIR, "dist", "media", "Img", "inicio.jpg")
        img = Image.open(img_path)
        img = img.resize((500, 400), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        
        img_label = tk.Label(frame, image=img_tk)
        img_label.image = img_tk
        img_label.pack(pady=20)
        
        btn_frame = tk.Frame(frame, bg="#f0f0f0")
        btn_frame.pack(pady=20)
        
        btn_registro = tk.Button(btn_frame, text="Registrarse", command=self.mostrar_pantalla_registro,
                               width=15, height=2, bg="#4CAF50", fg="white")
        btn_registro.pack(side="left", padx=10)
        
        btn_login = tk.Button(btn_frame, text="Iniciar Sesión", command=self.mostrar_pantalla_login,
                            width=15, height=2, bg="#2196F3", fg="white")
        btn_login.pack(side="left", padx=10)
    
    def mostrar_pantalla_registro(self):
        self.limpiar_pantalla()
        
        frame = tk.Frame(self.root, bg="#f0f0f0", padx=20, pady=20)
        frame.pack(expand=True, fill="both")
        
        label = tk.Label(frame, text="Registro de Nuevo Usuario", 
                        font=("Arial", 18), bg="#f0f0f0")
        label.pack(pady=20)
        
        form_frame = tk.Frame(frame, bg="#f0f0f0")
        form_frame.pack(pady=20)
        
        # Campos del formulario
        tk.Label(form_frame, text="Nombre:", bg="#f0f0f0").grid(row=0, column=0, sticky="e", pady=5)
        tk.Entry(form_frame, textvariable=self.nombre_var, width=30).grid(row=0, column=1, pady=5)
        
        tk.Label(form_frame, text="País:", bg="#f0f0f0").grid(row=1, column=0, sticky="e", pady=5)
        tk.Entry(form_frame, textvariable=self.pais_var, width=30).grid(row=1, column=1, pady=5)
        
        tk.Label(form_frame, text="Idioma:", bg="#f0f0f0").grid(row=2, column=0, sticky="e", pady=5)
        tk.Entry(form_frame, textvariable=self.idioma_var, width=30).grid(row=2, column=1, pady=5)
        
        # Botón para capturar foto
        btn_capturar = tk.Button(frame, text="Capturar Foto", command=self.capturar_foto_registro,
                                width=15, height=2, bg="#FF9800", fg="white")
        btn_capturar.pack(pady=20)
        
        # Botón para registrar
        btn_registrar = tk.Button(frame, text="Registrar", command=self.registrar_usuario,
                                 width=15, height=2, bg="#4CAF50", fg="white")
        btn_registrar.pack(pady=10)
        
        # Botón para volver
        btn_volver = tk.Button(frame, text="Volver", command=self.mostrar_pantalla_inicio,
                              width=15, height=2, bg="#9E9E9E", fg="white")
        btn_volver.pack(pady=10)
    
    def mostrar_pantalla_login(self):
        self.limpiar_pantalla()
        
        frame = tk.Frame(self.root, bg="#f0f0f0", padx=20, pady=20)
        frame.pack(expand=True, fill="both")
        
        label = tk.Label(frame, text="Iniciar Sesión con Reconocimiento Facial", 
                        font=("Arial", 18), bg="#f0f0f0")
        label.pack(pady=20)
        
        # Mostrar vista de cámara
        self.cam_label = tk.Label(frame, bg="black")
        self.cam_label.pack(pady=20)
        
        # Botón para iniciar reconocimiento
        btn_reconocer = tk.Button(frame, text="Iniciar Reconocimiento", command=self.iniciar_reconocimiento,
                                 width=20, height=2, bg="#2196F3", fg="white")
        btn_reconocer.pack(pady=20)
        
        # Botón para volver
        btn_volver = tk.Button(frame, text="Volver", command=self.mostrar_pantalla_inicio,
                              width=15, height=2, bg="#9E9E9E", fg="white")
        btn_volver.pack(pady=10)
        
        # Iniciar vista de cámara
        self.mostrar_camara()
    
    def limpiar_pantalla(self):
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def capturar_foto_registro(self):
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            self.cap = self.init_camera()
            if self.cap is None:
                messagebox.showerror("Error", "No se pudo abrir la cámara")
                return
        
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "No se pudo capturar la imagen de la cámara")
            return
        
        # El resto igual...
        top = tk.Toplevel(self.root)
        top.title("Foto Capturada")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((400, 300), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        
        label = tk.Label(top, image=img_tk)
        label.image = img_tk
        label.pack(padx=10, pady=10)
        
        btn_aceptar = tk.Button(top, text="Aceptar", command=lambda: self.guardar_foto(frame, top),
                            width=10, bg="#4CAF50", fg="white")
        btn_aceptar.pack(side="left", padx=10, pady=10)
        
        btn_reintentar = tk.Button(top, text="Reintentar", command=top.destroy,
                                width=10, bg="#F44336", fg="white")
        btn_reintentar.pack(side="right", padx=10, pady=10)

    
    def guardar_foto(self, frame, top):
        nombre = self.nombre_var.get().strip()
        if not nombre:
            messagebox.showerror("Error", "Debe ingresar un nombre")
            return
        
        # Crear carpeta si no existe
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        usuarios_folder = os.path.join(BASE_DIR, "dist", "media", "Usuarios")
        if not os.path.exists(usuarios_folder):
            os.makedirs(usuarios_folder)
        
        # Guardar imagen
        ruta_imagen = os.path.join(usuarios_folder, f"{nombre}.jpg")
        cv2.imwrite(ruta_imagen, frame)
        
        # Actualizar base de datos
        h, w, _ = frame.shape
        detector_caras.setInputSize((w, h))
        ret, caranueva = detector_caras.detect(frame)
        
        if ret and caranueva is not None:
            caracrop = fr.alignCrop(frame, caranueva[0])
            codcara = fr.feature(caracrop)
            
            usuarios[nombre] = {
                "pais": self.pais_var.get(),
                "idioma": self.idioma_var.get(),
                "vector": codcara,
                "imagen": frame.copy()
            }

            ruta_db = os.path.join(BASE_DIR,"dist" ,"usuarios.db")
            conn = sqlite3.connect(ruta_db)
            cursor = conn.cursor()

            # Asegúrate de que la tabla tiene columna imagen (si vas a usarla)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS usuarios (
                    nombre TEXT PRIMARY KEY,
                    pais TEXT,
                    idioma TEXT
                )
            ''')

            cursor.execute('''
                INSERT OR REPLACE INTO usuarios (nombre, pais, idioma)
                VALUES (?, ?, ?)
            ''', (nombre, self.pais_var.get(), self.idioma_var.get()))

            conn.commit()
            conn.close()
            
            messagebox.showinfo("Éxito", "Usuario registrado correctamente")
            top.destroy()
            self.mostrar_pantalla_inicio()
        else:
            messagebox.showerror("Error", "No se detectó un rostro en la imagen")
            os.remove(ruta_imagen)

    def cerrar_aplicacion(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()
    
    def mostrar_camara(self):
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            self.cap = self.init_camera()
            if self.cap is None:
                return
        
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            
            self.cam_label.configure(image=img_tk)
            self.cam_label.image = img_tk
            
        self.root.after(10, self.mostrar_camara)

    
    def iniciar_reconocimiento(self):
        global nombre, reconocido
        print("🟢 Se ha hecho clic en 'Iniciar Reconocimiento'")
        reconocido = False

        if not hasattr(self, 'cap') or not self.cap.isOpened():
            print("⚠️ Cámara no abierta. Reintentando...")
            self.cap = self.init_camera()
            if self.cap is None:
                messagebox.showerror("Error", "No se pudo abrir la cámara")
                return

        ret, frame = self.cap.read()
        if not ret:
            print("❌ No se pudo leer frame de la cámara")
            messagebox.showerror("Error", "No se pudo capturar imagen de la cámara")
            return
        
        h, w = frame.shape[:2]
        print(f"📏 Tamaño del frame: {w}x{h}")
        detector_caras.setInputSize((w, h))
        
        # Detectar caras
        ret, caras = detector_caras.detect(frame)
        if ret and caras is not None:
            for cara in caras:
                c = cara.astype(int)
                try:
                    caracrop = fr.alignCrop(frame, cara)
                    codcara = fr.feature(caracrop)
                except:
                    continue

                maximo = -999
                nombre = "Desconocido"

                for usuario, info in usuarios.items():
                    if info["vector"] is not None:
                        semejanza = fr.match(info["vector"], codcara, cv2.FaceRecognizerSF_FR_COSINE)
                        if semejanza > maximo:
                            maximo = semejanza
                            nombre = usuario

                if maximo >= 0.5:
                    reconocido = True
                    messagebox.showinfo("Bienvenido", f"¡Bienvenido {nombre}!")

                    datos = obtener_info_usuario(nombre)
                    idiomaU = datos["idioma"].lower()
                    pais = datos["pais"].lower()
                    usuarios[nombre]["pais"] = pais
                    usuarios[nombre]["idioma"] = idiomaU
                    
                    self.mostrar_pantalla_principal(nombre)
                    print (usuarios)
                    return
        
        messagebox.showwarning("No reconocido", "No se reconoció ningún usuario registrado")

    def getUsuarioActivo(self):
        return self.nombre_usuario

    
    def mostrar_pantalla_principal(self, nombre_usuario):
         
        self.limpiar_pantalla()
        
        frame = tk.Frame(self.root, bg="#f0f0f0", padx=20, pady=20)
        frame.pack(expand=True, fill="both")
        
        # Mostrar información del usuario
        info_usuario = usuarios.get(nombre_usuario, {})
      
        
        header_frame = tk.Frame(frame, bg="#f0f0f0")
        header_frame.pack(pady=20)
        
        # Mostrar imagen del usuario
        img_usuario = info_usuario.get("imagen", None)
        if img_usuario is not None:
            img_usuario = cv2.cvtColor(img_usuario, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_usuario)
            img = img.resize((100, 100), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            
            img_label = tk.Label(header_frame, image=img_tk, bg="#f0f0f0")
            img_label.image = img_tk
            img_label.pack(side="left", padx=10)
        
        # Mostrar datos del usuario
        datos_frame = tk.Frame(header_frame, bg="#f0f0f0")
        datos_frame.pack(side="left", padx=10)
        
        tk.Label(datos_frame, text=f"Usuario: {nombre_usuario}", 
                font=("Arial", 14), bg="#f0f0f0").pack(anchor="w")
        tk.Label(datos_frame, text=f"País: {info_usuario.get('pais', 'Desconocido')}", 
                bg="#f0f0f0").pack(anchor="w")
        tk.Label(datos_frame, text=f"Idioma: {info_usuario.get('idioma', 'Desconocido')}", 
                bg="#f0f0f0").pack(anchor="w")
        
        # Botón para iniciar realidad aumentada
        btn_ra = tk.Button(frame, text="Iniciar Realidad Aumentada", 
                          command=lambda: self.iniciar_realidad_aumentada(nombre_usuario),
                          width=25, height=2, bg="#2196F3", fg="white")
        btn_ra.pack(pady=20)
        
        # Botón para cerrar sesión
        btn_cerrar = tk.Button(frame, text="Cerrar Sesión", command=self.mostrar_pantalla_inicio,
                             width=15, height=2, bg="#F44336", fg="white")
        btn_cerrar.pack(pady=10)
    
    def iniciar_realidad_aumentada(self, nombre_usuario):
        self.limpiar_pantalla()
        
        frame = tk.Frame(self.root)
        frame.pack(expand=True, fill="both")
        
        # Etiqueta para mostrar la cámara con RA
        self.ra_label = tk.Label(frame)
        self.ra_label.pack(expand=True, fill="both")
        
        # Botón para volver
        btn_volver = tk.Button(frame, text="Volver", 
                              command=lambda: self.mostrar_pantalla_principal(nombre_usuario),
                              width=15, height=2, bg="#9E9E9E", fg="white")
        btn_volver.pack(pady=10)
        
        # Iniciar bucle de realidad aumentada
        self.mostrar_realidad_aumentada(nombre_usuario)
    
    
    
    def mostrar_realidad_aumentada(self, nombre_usuario):
        global elimina

        if elimina:
            self.eliminar_usuario_y_cerrar_app(nombre_usuario)

        if not hasattr(self, 'cap') or not self.cap.isOpened():
            self.cap = self.init_camera()
            if self.cap is None:
                messagebox.showerror("Error", "No se pudo abrir la cámara")
                return
        
        
        ret, frame = self.cap.read()
       
        if not ret:
            messagebox.showerror("Error", "No se pudo capturar la imagen de la cámara")
            return
        if ret:
            frame_ra = realidadMixta(frame,self.ancho, self.alto)
            
            # Obtener la imagen del usuario y crear la máscara circular
        img_user = usuarios.get(nombre_usuario, {}).get("imagen", None)
        if img_user is not None:
            # Redimensionar a tamaño más pequeño (70x70)
            thumb = cv2.resize(img_user, (70, 70))

            # Crear máscara circular
            mask = np.zeros((70, 70), dtype=np.uint8)
            cv2.circle(mask, (35, 35), 35, 255, -1)

            # Aplicar máscara
            thumb_circle = cv2.bitwise_and(thumb, thumb, mask=mask)

            # Coordenadas para esquina superior derecha
            h_frame, w_frame, _ = frame.shape
            x_offset = w_frame - 80
            y_offset = 30

            # Nombre justo encima
            cv2.putText(frame, f"{nombre_usuario}", (x_offset , y_offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Mostrar imagen circular
            roi = frame[y_offset:y_offset+70, x_offset:x_offset+70]
            roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
            final = cv2.add(roi_bg, thumb_circle)
            frame[y_offset:y_offset+70, x_offset:x_offset+70] = final
            
            frame_rgb = cv2.cvtColor(frame_ra, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((800, 600), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            
            self.ra_label.configure(image=img_tk)
            self.ra_label.image = img_tk
        
        self.root.after(10, lambda: self.mostrar_realidad_aumentada(nombre_usuario))
    
    def registrar_usuario(self):
        nombre = self.nombre_var.get().strip()
        pais = self.pais_var.get().strip()
        idioma = self.idioma_var.get().strip()
        
        if not nombre or not pais or not idioma:
            messagebox.showerror("Error", "Todos los campos son obligatorios")
            return
        
        if nombre in usuarios:
            messagebox.showerror("Error", "El nombre de usuario ya existe")
            return
        
        # Verificar que se haya capturado una foto
        if "imagen" not in usuarios.get(nombre, {}):
            messagebox.showerror("Error", "Debe capturar una foto para registrarse")
            return
        
        messagebox.showinfo("Éxito", "Usuario registrado correctamente")
        self.mostrar_pantalla_inicio()

    def configurar_camara_y_modelos(self):
        """Configura los parámetros de la cámara y carga los modelos 3D"""
        global cameraMatrix, distCoeffs, modelos, escenas

        # Obtener ancho y alto desde la cámara
        self.ancho = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.alto = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        try:
            if myCam == 0:
                import camara0
                print("Utilizando cámara 0")
                self.cameraMatrix = camara0.cameraMatrix
                self.distCoeffs = camara0.distCoeffs
            elif myCam == 1:
                import camara
                print("Utilizando cámara 1")
                self.cameraMatrix = camara.cameraMatrix
                self.distCoeffs = camara.distCoeffs
        except ImportError:
            self.cameraMatrix = np.array([
                [1000, 0, self.ancho / 2],
                [0, 1000, self.alto / 2],
                [0, 0, 1]
            ])
            self.distCoeffs = np.zeros((5, 1))

        # Cargar modelos 3D
        self.modelos = {}
        self.escenas = {}

        for id_aruco, ruta in modelos_por_id.items():
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(BASE_DIR, "dist", "media", "Models", ruta)
            m = cuia.modeloGLTF(path)
            m.rotar((np.pi / 2.0, 0, 0))
            m.escalar(0.3)
            m.flotar()
            
            anims = m.animaciones()
            if len(anims) > 0:
                m.animar(anims[0])

            escena = cuia.escenaPYGFX(fov(self.cameraMatrix, self.ancho, self.alto), self.ancho, self.alto)
            escena.agregar_modelo(m)
            escena.ilumina_modelo(m)
            escena.iluminar()

            self.modelos[id_aruco] = m
            self.escenas[id_aruco] = escena

        # Hacer disponibles las variables globalmente
        cameraMatrix = self.cameraMatrix
        distCoeffs = self.distCoeffs
        modelos = self.modelos
        escenas = self.escenas



# ===========================================
# INICIAR APLICACIÓN
# ===========================================
if __name__ == "__main__":
    root_selector = tk.Tk()
    selector = SelectorDispositivos(root_selector)
    root_selector.mainloop()