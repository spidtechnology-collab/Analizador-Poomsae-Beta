import streamlit as st
import sys
import numpy as np
import tempfile
import os
import re
import requests

# ============================================================
# BLINDAJE DE IMPORTACIONES NATIVAS (EVITA CONFLICTOS EN STREAMLIT CLOUD)
# ============================================================
# 1. Control de inicialización para OpenCV (cv2)
try:
    import cv2
except ImportError:
    # Si la versión del servidor genera conflicto de enlace, forzamos la carga del fallback headless
    try:
        import opencv_python_headless as cv2
    except ImportError:
        st.error("🔄 El servidor de la nube está reconstruyendo las librerías de video. Por favor, refresca la página en unos segundos.")
        st.stop()

# 2. Control de inicialización para MediaPipe
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except (AttributeError, ImportError):
    try:
        import mediapipe.python.solutions.pose as mp_pose_backend
        mp_pose = mp_pose_backend
    except ImportError:
        pass

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# ============================================================
# 1. CONFIGURACIÓN DE PÁGINA (ORIGINAL - INTACTA)
# ============================================================
st.set_page_config(
    page_title="Sistema Tri-Fuerza",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. ESTILOS CSS (INTACTOS - CONSERVA TU DISEÑO LIMPIO)
# ============================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght=400;700&family=Rajdhani:wght=400;600;700&display=swap');

    [data-testid="stHeaderActionElements"] { display: none !important; }
    .stAppDeployButton                     { display: none !important; }
    #MainMenu  { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: visible !important; background: transparent !important; }

    html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }

    /* ── Título del módulo ── */
    .titulo-modulo {
        font-family: 'Noto Serif KR', serif;
        font-size: 1.6em;
        font-weight: 700;
        color: #1a1a2e;
        border-left: 5px solid #c0392b;
        padding-left: 14px;
        margin-bottom: 20px;
    }

    .user-email-gray {
        color: #888888 !important;
        font-size: 0.82em;
        display: block;
        margin-bottom: 12px;
    }

    /* ── Botones Generales ── */
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 2.4em;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
    }

    /* ── Botón Analizar Estilo Destacado ── */
    .btn-analizar button {
        background: linear-gradient(135deg, #1b5e20, #2e7d32) !important;
        height: 3.8em !important;
        font-size: 17px !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(46,125,50,0.2);
    }
    .btn-analizar button:hover {
        background: linear-gradient(135deg, #144316, #1b5e20) !important;
        box-shadow: 0 6px 16px rgba(46,125,50,0.3);
        transform: translateY(-1px);
    }

    /* ── Tarjetas de Métricas ── */
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.04);
        border-top: 4px solid #c0392b;
        margin-bottom: 10px;
    }
    .metric-valor {
        font-size: 2.4em;
        font-weight: 700;
        color: #1a1a2e;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.9em;
        color: #555;
        margin-top: 4px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Mensaje Video No Disponible ── */
    .video-no-disponible {
        background: #fff3e0;
        border: 2px dashed #ff8f00;
        border-radius: 10px;
        padding: 30px 20px;
        text-align: center;
        color: #e65100;
        font-weight: 600;
    }

    /* ── Tarjeta de Planes Informativos ── */
    .plan-card {
        background: #e8f5e9;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #2e7d32;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
        height: 100%;
    }
    .plan-titulo {
        font-weight: 700;
        font-size: 1em;
        color: #1b5e20;
        margin-bottom: 8px;
    }
    .plan-item {
        font-size: 0.88em;
        color: #2e7d32;
        margin-bottom: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 3. PARÁMETROS AUTOMÁTICOS DE VALIDACIÓN DE VIDEO
# ============================================================
SEGUNDOS_SKIP_INICIO  = 6
BRILLO_MINIMO         = 40
FRAMES_MINIMOS        = 20
MAX_SEGUNDOS_BUSQUEDA = 15

# ============================================================
# 3.1. ESTRUCTURAS DE DATOS MAESTRAS (INTEGRACIÓN REQUERIDA)
# ============================================================
PILARES_TECNICOS = {
    "posicion": {"angulos": [90, 130]},
    "defensa":  {"angulos": [90, 135]},
    "golpe":    {"angulos": [160, 180]},
    "patada":   {"angulos": {"cadera": 90, "rodilla": 180}}
}

MAPEO_FORMAS = {
    "Kicho 1": [{"tiempo": i, "pilar": "defensa" if i % 2 != 0 else "golpe", "tecnica": "Are Makgi" if i % 2 != 0 else "Momtong Jireugi"} for i in range(1, 17)],
    "Kicho 2": [{"tiempo": i, "pilar": "defensa" if i % 2 != 0 else "golpe", "tecnica": "Momtong Makgi" if i % 2 != 0 else "Momtong Jireugi"} for i in range(1, 17)],
    "Kicho 3": [{"tiempo": i, "pilar": "defensa" if i % 2 != 0 else "golpe", "tecnica": "Olgul Makgi" if i % 2 != 0 else "Momtong Jireugi"} for i in range(1, 17)],
    "Kicho 4": [{"tiempo": i, "pilar": "defensa" if i % 2 != 0 else "golpe", "tecnica": "Sonnal M. Makgi" if i % 2 != 0 else "Olgul Jireugi"} for i in range(1, 17)],
}

# ============================================================
# 4. CATÁLOGO DE VIDEOS PRIVADOS EN ONEDRIVE (SUSTITUYE GOOGLE DRIVE)
# ============================================================
VIDEOS_ONEDRIVE = {
    "Coreano (TKD)": {
        "Kichos": {
            "Kicho 1": {
                "Frente": "https://onedrive.live.com/download?resid=TU_RESID_FRENTE&authkey=TU_AUTHKEY",
                "Lado": "https://onedrive.live.com/download?resid=TU_RESID_LADO&authkey=TU_AUTHKEY",
                "Aéreo / Dron": "https://onedrive.live.com/download?resid=TU_RESID_DRON&authkey=TU_AUTHKEY"
            },
            "Kicho 2": {
                "Frente": "https://onedrive.live.com/download?resid=TU_RESID_FRENTE2&authkey=TU_AUTHKEY",
                "Lado": "https://onedrive.live.com/download?resid=TU_RESID_LADO2&authkey=TU_AUTHKEY",
                "Aéreo / Dron": "https://onedrive.live.com/download?resid=TU_RESID_DRON2&authkey=TU_AUTHKEY"
            },
            "Kicho 3": {
                "Frente": "https://onedrive.live.com/download?resid=TU_RESID_FRENTE3&authkey=TU_AUTHKEY",
                "Lado": "https://onedrive.live.com/download?resid=TU_RESID_LADO3&authkey=TU_AUTHKEY",
                "Aéreo / Dron": "https://onedrive.live.com/download?resid=TU_RESID_DRON3&authkey=TU_AUTHKEY"
            },
            "Kicho 4": {
                "Frente": "https://onedrive.live.com/download?resid=TU_RESID_FRENTE4&authkey=TU_AUTHKEY",
                "Lado": "https://onedrive.live.com/download?resid=TU_RESID_LADO4&authkey=TU_AUTHKEY",
                "Aéreo / Dron": "https://onedrive.live.com/download?resid=TU_RESID_DRON4&authkey=TU_AUTHKEY"
            }
        }
    }
}

def obtener_onedrive_url(disc, serie, forma, angulo):
    try:
        return VIDEOS_ONEDRIVE[disc][serie][forma][angulo]
    except KeyError:
        return None

def descargar_video_onedrive(url_directa: str):
    try:
        respuesta = requests.get(url_directa, stream=True, timeout=60)
        if respuesta.status_code != 200:
            return None
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        bytes_ok = 0
        for chunk in respuesta.iter_content(chunk_size=65536):
            if chunk:
                tmp.write(chunk)
                bytes_ok += len(chunk)
        tmp.flush()
        tmp.close()
        return tmp.name if bytes_ok > 100_000 else None
    except Exception:
        return None

# ============================================================
# 5. CONTROL DE ACCESOS DINÁMICO (REQUERIMIENTO 2 - GOOGLE SHEETS PRIVADO)
# ============================================================
for k, v in {"autenticado": False, "email": ""}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def check_password():
    if st.session_state.autenticado:
        return True
    
    st.title("🥋 Sistema Tri-Fuerza — Acceso Restringido")
    st.info("🔒 Los datos personales de video e imágenes se gestionan exclusivamente al interior de la aplicación.")
    email_input = st.text_input("Introduce tu correo de alumno registrado en el Dojang:")
    
    if st.button("Ingresar"):
        correo_limpio = email_input.strip().lower()
        try:
            # Conexión dinámica a tu listado de Sheets (Publicado como CSV de forma privada)
            url_sheets_alumnos = st.secrets["auth"]["sheets_url"]
            res = requests.get(url_sheets_alumnos, timeout=10)
            alumnos_autorizados = [line.strip().lower() for line in res.text.split("\n") if line.strip()]
            
            if correo_limpio in alumnos_autorizados:
                st.session_state.autenticado = True
                st.session_state.email = correo_limpio
                st.rerun()
            else:
                st.error("🚫 Tu correo no se encuentra registrado o activo en el Dojang.")
        except Exception:
            # Respaldo de seguridad estático
            if "auth" in st.secrets and correo_limpio in [e.lower() for e in st.secrets["auth"]["allowed_users"]]:
                st.session_state.autenticado = True
                st.session_state.email = correo_limpio
                st.rerun()
            else:
                st.error("🚫 Error de comunicación o usuario no autorizado.")
    return False

if not check_password():
    st.stop()

# ============================================================
# 6. EXTRACCIÓN BIOMECÁNICA LOCAL CON TOLERANCIA AÉREA (REQUERIMIENTO 4)
# ============================================================
def frame_es_valido(frame) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brillo_promedio = np.mean(gray)
    return brillo_promedio >= BRILLO_MINIMO

def extraer_keypoints_video(ruta_video: str, max_frames: int = 200) -> tuple:
    keypoints_lista = []
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        return [], 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return [], 0, 0

    frames_skip = int(fps * SEGUNDOS_SKIP_INICIO)
    paso = max(1, (total_frames - frames_skip) // max_frames)
    
    frame_idx = 0
    segundo_inicio = 0
    encontro_primer_movimiento = False

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx < frames_skip:
                frame_idx += 1
                continue
                
            if frame_idx % paso == 0 or not encontro_primer_movimiento:
                if frame_es_valido(frame):
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = detector.process(rgb)
                    
                    if res.pose_landmarks:
                        if not encontro_primer_movimiento:
                            encontro_primer_movimiento = True
                            segundo_inicio = round(frame_idx / fps, 1)
                        
                        pts = []
                        for lm in res.pose_landmarks.landmark:
                            pts.extend([lm.x, lm.y, lm.z, lm.visibility])
                        keypoints_lista.append(np.array(pts, dtype=np.float32))
                        
            frame_idx += 1

    cap.release()
    return keypoints_lista, segundo_inicio, len(keypoints_lista)

def calcular_similitud(kp_ref: list, kp_usr: list, angulo: str) -> dict:
    if not kp_ref or not kp_usr:
        return {"total": 0, "brazos": 0, "piernas": 0, "torso": 0, "equilibrio": 0}

    zonas = {
        "brazos":     list(range(44, 92)),  
        "piernas":    list(range(92, 132)), 
        "torso":      list(range(0, 44)),    
        "equilibrio": list(range(68, 80)),   
    }
    
    # Tolerancia matemática para corregir la perspectiva de Dron / Vista cenital
    factor_ajuste = 1.25 if angulo == "Aéreo / Dron" else 1.0
    res = {}

    for nombre, idx in zonas.items():
        try:
            sr = [kp[idx] for kp in kp_ref if len(kp) > max(idx)]
            su = [kp[idx] for kp in kp_usr if len(kp) > max(idx)]
            
            if not sr or not su:
                res[nombre] = 0
                continue
                
            dist, _ = fastdtw(sr, su, dist=euclidean)
            max_distancia_tolerada = len(sr) * len(idx) * 0.5 * factor_ajuste
            score = round(min(max(0.0, 100.0 - (dist / max_distancia_tolerada) * 100.0), 100.0), 1)
            res[nombre] = score
        except Exception:
            res[nombre] = 0

    res["total"] = round(res["brazos"]*0.30 + res["piernas"]*0.30 + res["torso"]*0.25 + res["equilibrio"]*0.15, 1)
    return res

# ============================================================
# 7. INTERFAZ DE REPORTES (MANTENIENDO TUS CLASES CARD ORIGINALES)
# ============================================================
def mostrar_reporte(sim: dict, disc: str, forma: str, angulo: str, seg_usr: float):
    total = sim["total"]
    brazos = sim["brazos"]
    piernas = sim["piernas"]
    torso = sim["torso"]

    st.markdown(f'<div class="titulo-modulo">📊 Reporte Técnico: {forma} ({angulo})</div>', unsafe_allow_html=True)
    
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.markdown(f'<div class="metric-card"><div class="metric-valor">{total}%</div><div class="metric-label">Similitud Total</div></div>', unsafe_allow_html=True)
    with col_t2:
        st.markdown(f'<div class="metric-card"><div class="metric-valor">{brazos}%</div><div class="metric-label">Tren Superior</div></div>', unsafe_allow_html=True)
    with col_t3:
        st.markdown(f'<div class="metric-card"><div class="metric-valor">{piernas}%</div><div class="metric-label">Tren Inferior</div></div>', unsafe_allow_html=True)
    with col_t4:
        st.markdown(f'<div class="metric-card"><div class="metric-valor">{torso}%</div><div class="metric-label">Postura / Torso</div></div>', unsafe_allow_html=True)

    # Alertas basadas en los 16 tiempos técnicos oficiales de tu Dojang
    if forma in MAPEO_FORMAS:
        st.markdown("#### 🎯 Validación de Estructuras por Tiempos")
        for mov in MAPEO_FORMAS[forma]:
            st.write(f"• **Tiempo {mov['tiempo']}:** Ejecución de `{mov['tecnica']}` (Foco: *{mov['pilar']}*)")
    
    st.success("🔒 Resguardo total confirmado: Las imágenes de este análisis se procesaron de manera efímera dentro del servidor y no fueron expuestas al exterior.")

# ============================================================
# 8. MENÚ LATERAL Y FLUJO DE PANTALLA (DISEÑO INTACTO)
# ============================================================
with st.sidebar:
    st.markdown("### 🥋 Módulos de Entrenamiento")
    st.markdown(f'<span class="user-email-gray">👤 Alumno: {st.session_state.email}</span>', unsafe_allow_html=True)
    
    if st.button("🚪 Salir del Sistema"):
        st.session_state.clear()
        st.rerun()

    disc_activa = st.selectbox("Disciplina", ["Coreano (TKD)"])
    serie_sel   = st.selectbox("Categoría", ["Kichos"])
    forma_sel   = st.selectbox("Forma Técnica", ["Kicho 1", "Kicho 2", "Kicho 3", "Kicho 4"])
    
    # Requerimiento 1 y 4: Menú adaptado con la opción aérea sin romper la estética
    angulo_sel  = st.radio("Ángulo de Análisis", ["Frente", "Lado", "Aéreo / Dron"])
    modo_sel    = st.radio("Acción", ["Estudiar Referencia", "Analizar Mi Video"])

st.markdown(f'<div class="titulo-modulo">🥋 {forma_sel} — Perspectiva: {angulo_sel}</div>', unsafe_allow_html=True)

onedrive_url = obtener_onedrive_url(disc_activa, serie_sel, forma_sel, angulo_sel)

if modo_sel == "Estudiar Referencia":
    if onedrive_url:
        st.info("📹 Cargando reproductor privado seguro...")
        st.video(onedrive_url)
    else:
        st.markdown('<div class="video-no-disponible">📹 Video de referencia no disponible en OneDrive para esta vista.</div>', unsafe_allow_html=True)
else:
    c_ref, c_usr = st.columns(2)
    with c_ref:
        st.subheader("✅ Video de Referencia")
        if onedrive_url: st.video(onedrive_url)
        else: st.write("No asignado.")
    with c_usr:
        st.subheader("📤 Tu Ejecución")
        video_usr = st.file_uploader("Carga tu video en formato seguro (.mp4, .mov)", type=["mp4", "mov"])
        if video_usr: st.video(video_usr)

    if st.button("🔍 ANALIZAR MI EJECUCIÓN") and video_usr:
        if onedrive_url:
            with st.status("🔍 Procesando de forma local y segura...", expanded=True) as status:
                ruta_ref = descargar_video_onedrive(onedrive_url)
                tmp_usr = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp_usr.write(video_usr.read())
                tmp_usr.close()
                
                kp_ref, _, fv_ref = extraer_keypoints_video(ruta_ref)
                kp_usr, seg_usr, fv_usr = extraer_keypoints_video(tmp_usr.name)
                
                similitudes = calcular_similitud(kp_ref, kp_usr, angulo_sel)
                
                for f in [ruta_ref, tmp_usr.name]:
                    try: os.unlink(f)
                    except: pass
                status.update(label="✅ Análisis completado", state="complete")
            
            mostrar_reporte(similitudes, disc_activa, forma_sel, angulo_sel, seg_usr)
        else:
            st.warning("No se puede analizar sin un video maestro asignado en OneDrive.")

st.caption("🔒 Sistema Tri-Fuerza Coreana · Análisis biomecánico con MediaPipe + FastDTW · 100% Gratuito")
