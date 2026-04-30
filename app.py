import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import re
import requests
import mediapipe as mp
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

mp_pose = mp.solutions.pose

# ============================================================
# 1. CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Sistema Tri-Fuerza",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. ESTILOS CSS
# ============================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@400;700&family=Rajdhani:wght@400;600;700&display=swap');

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

    /* ── Sidebar ── */
    .user-email-gray {
        color: #888888 !important;
        font-size: 0.82em;
        display: block;
        margin-bottom: 12px;
    }

    /* ── Botones generales ── */
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 2.4em;
        font-size: 13px;
        font-weight: 700;
        font-family: 'Rajdhani', sans-serif;
        letter-spacing: 0.5px;
    }

    /* ── Botón ANALIZAR ── */
    .btn-analizar button {
        background: linear-gradient(135deg, #1b5e20, #2e7d32) !important;
        height: 3.8em !important;
        font-size: 17px !important;
        color: white !important;
        border: none !important;
        letter-spacing: 1px;
    }

    /* ── Tarjetas de métricas ── */
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border-top: 4px solid #c0392b;
        margin-bottom: 10px;
    }
    .metric-valor { font-size: 2.4em; font-weight: 700; }
    .metric-label { font-size: 0.9em; color: #555; font-weight: 600; }

    /* ── Video no disponible ── */
    .video-no-disponible {
        background: #fff3e0;
        border: 2px dashed #ff8f00;
        border-radius: 10px;
        padding: 30px 20px;
        text-align: center;
        color: #e65100;
        font-weight: 600;
        font-size: 1em;
    }

    /* ── Tarjeta de plan de práctica ── */
    .plan-card {
        background: #e8f5e9;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #2e7d32;
        height: 100%;
    }
    .plan-titulo {
        font-weight: 700;
        font-size: 1em;
        color: #1b5e20;
        margin-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 3. PARÁMETROS INTERNOS DE VALIDACIÓN
# ============================================================
SEGUNDOS_SKIP_INICIO  = 6
BRILLO_MINIMO         = 40
FRAMES_MINIMOS        = 20
MAX_SEGUNDOS_BUSQUEDA = 15

# ============================================================
# 4. CATÁLOGO DE VIDEOS EN GOOGLE DRIVE
# ============================================================
VIDEOS_DRIVE = {
    "Taekwondo": {
        "Kichos": {
            "Kicho 1": {
                "Frente": "1Vydz8f0UlAOm2gu5L77p-HZ85OuoXC67",
                "Lado":   "1KqtzgHpMDSa5CYwKvNiCmcqXQ6ib3Rxv",
            },
            "Kicho 2": {
                "Frente": "1_FvliAlgfhzsYSNuU8PWnb7uhzxM7xNr",
                "Lado":   "1oEYwsvC5cyqef_uYAX9AXLv214JitqlQ",
            },
            "Kicho 3": {
                "Frente": "1H4CWN0criUxXNVazG8M0HVeJzZ6I2t_a",
                "Lado":   "1PEyqqI38oSUQDWg71Tvz3pnWGgic7GrA",
            },
            "Kicho 4": {
                "Frente": "1pTZsmrY-G2P68tsIJkOz9bevlx_s8Iyz",
                "Lado":   "1PUCzNuHKcfGyWqaxbF0BVkdrDYL_Ixty",
            },
        },
        "Palgwes": {
            "Palgwe 1": {
                "Frente": "1jDUitMc6MttrNC9tjhMuWz69pqB-U2Yr",
                "Lado":   "1ZXWqKcA3Prpka-P8s5cPVqp-_4xa9V3w",
            },
            "Palgwe 2": {
                "Frente": "1cb1H_rzCdPC2PdZn63-jGztZwDFRR4gf",
                "Lado":   "1D5c1IfZ7F9hJzxDcAMKlFdq7uRvxojhf",
            },
            "Palgwe 3": {
                "Frente": "1K2GrdZos6L0ALdWgrrh7N5SvnGwuOrjj",
                "Lado":   "1TGeLDd9dRHXl6N3_r1Frr1bUAWCCbd0h",
            },
            "Palgwe 4": {
                "Frente": "1_abxssBy7bAIUn63_V2BtZqJEYns4Lc3",
                "Lado":   "18NWA1zYMvB5fHw_N-TexONsrptubCKcj",
            },
        },
        "Taegeuks": {},   # Próximamente
        "Judanyas": {},   # Próximamente
    },
    "Hapkido":       {},  # Próximamente
    "Haidong Gumdo": {},  # Próximamente
}

def obtener_drive_id(disciplina, serie, forma, angulo):
    try:
        return VIDEOS_DRIVE[disciplina][serie][forma][angulo]
    except KeyError:
        return None

def descargar_video_drive(drive_id: str):
    sesion   = requests.Session()
    url_base = "https://drive.google.com/uc"
    params   = {"export": "download", "id": drive_id}
    try:
        respuesta = sesion.get(url_base, params=params, stream=True, timeout=30)
        token = None
        for key, value in respuesta.cookies.items():
            if "download_warning" in key:
                token = value
                break
        if token is None:
            contenido = respuesta.content.decode("utf-8", errors="ignore")
            m = re.search(r'confirm=([0-9A-Za-z_\-]+)', contenido)
            if m:
                token = m.group(1)
            m2 = re.search(r'"([^"]+/uc\?export=download[^"]+)"', contenido)
            if m2:
                url_d = m2.group(1).replace("\\u003d","=").replace("\\u0026","&")
                respuesta = sesion.get(url_d, stream=True, timeout=120)
                token = "DIRECTO"
        if token and token != "DIRECTO":
            params["confirm"] = token
            respuesta = sesion.get(url_base, params=params, stream=True, timeout=120)
        if "text/html" in respuesta.headers.get("Content-Type",""):
            respuesta = sesion.get(
                f"https://drive.google.com/uc?id={drive_id}&export=download&confirm=t",
                stream=True, timeout=120
            )
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        bytes_ok = 0
        for chunk in respuesta.iter_content(chunk_size=65536):
            if chunk:
                tmp.write(chunk)
                bytes_ok += len(chunk)
        tmp.flush()
        tmp.close()
        if bytes_ok < 100_000:
            os.unlink(tmp.name)
            return None
        return tmp.name
    except Exception as e:
        st.error(f"❌ Error descargando desde Drive: {str(e)}")
        return None

# ============================================================
# 5. INICIALIZACIÓN DE ESTADOS
# ============================================================
for k, v in {"autenticado": False, "mostrar_ayuda": False, "email": ""}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# 6. AUTENTICACIÓN
# ============================================================
def check_password():
    if st.session_state.autenticado:
        return True
    st.title("🔐 Acceso — Sistema Tri-Fuerza Coreana")
    email_input = st.text_input("Introduce tu correo autorizado:")
    if st.button("Ingresar"):
        if "auth" in st.secrets:
            permitidos = st.secrets["auth"]["allowed_users"]
            if email_input.strip().lower() in [e.lower() for e in permitidos]:
                st.session_state.autenticado = True
                st.session_state.email = email_input.strip()
                st.rerun()
            else:
                st.error("🚫 Correo no autorizado.")
        else:
            st.warning("⚠️ Modo desarrollo — sin Secrets configurados.")
            st.session_state.autenticado = True
            st.session_state.email = email_input.strip() or "desarrollo@local"
            st.rerun()
    return False

if not check_password():
    st.stop()

# ============================================================
# 7. EXTRACCIÓN DE KEYPOINTS CON VALIDACIÓN AUTOMÁTICA
# ============================================================
def frame_es_valido(frame) -> bool:
    return np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) >= BRILLO_MINIMO

def extraer_keypoints_video(ruta_video: str, max_frames: int = 200) -> tuple:
    keypoints_lista = []
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        return [], 0, 0
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return [], 0, 0

    frames_skip    = int(fps * SEGUNDOS_SKIP_INICIO)
    frames_max_bus = int(fps * MAX_SEGUNDOS_BUSQUEDA)
    paso           = max(1, (total_frames - frames_skip) // max_frames)
    frame_idx      = 0
    segundo_inicio = 0
    encontro       = False

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx < frames_skip:
                frame_idx += 1
                continue
            if not frame_es_valido(frame):
                frame_idx += 1
                if frame_idx < frames_skip + frames_max_bus:
                    continue
                else:
                    break
            if frame_idx % paso == 0 or not encontro:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    res = detector.process(rgb)
                    rgb.flags.writeable = True
                    if res.pose_landmarks:
                        if not encontro:
                            encontro       = True
                            segundo_inicio = round(frame_idx / fps, 1)
                        pts = []
                        for lm in res.pose_landmarks.landmark:
                            pts.extend([lm.x, lm.y, lm.z, lm.visibility])
                        keypoints_lista.append(np.array(pts, dtype=np.float32))
                except Exception:
                    pass
            frame_idx += 1

    cap.release()
    return keypoints_lista, segundo_inicio, len(keypoints_lista)

# ============================================================
# 8. CÁLCULO DE SIMILITUD
# ============================================================
def calcular_similitud(kp_ref: list, kp_usr: list) -> dict:
    if not kp_ref or not kp_usr:
        return {k: 0 for k in ["total","brazos","piernas","torso","hombros","caderas","equilibrio"]}

    zonas = {
        "brazos":     list(range(44, 92)),
        "piernas":    list(range(92, 132)),
        "torso":      list(range(0,  44)),
        "hombros":    list(range(44, 56)),
        "caderas":    list(range(92, 100)),
        "equilibrio": list(range(68, 80)),
    }
    res = {}
    for nombre, idx in zonas.items():
        try:
            sr = [kp[idx] for kp in kp_ref if len(kp) > max(idx)]
            su = [kp[idx] for kp in kp_usr if len(kp) > max(idx)]
            if not sr or not su:
                res[nombre] = 0
                continue
            dist, _ = fastdtw(sr, su, dist=euclidean)
            max_d   = len(sr) * len(idx) * 0.5
            res[nombre] = round(min(max(0.0, 100.0 - (dist/max_d)*100.0), 100.0), 1)
        except Exception:
            res[nombre] = 0

    res["total"] = round(
        res["brazos"]     * 0.30 +
        res["piernas"]    * 0.30 +
        res["torso"]      * 0.25 +
        res["equilibrio"] * 0.15, 1
    )
    return res

# ============================================================
# 9. HELPERS DE REPORTE
# ============================================================
def color_score(s):
    return "#1b5e20" if s >= 80 else ("#e65100" if s >= 60 else "#c0392b")

def nivel_texto(s):
    if s >= 85: return "✅ Excelente"
    if s >= 70: return "🟡 Bueno"
    if s >= 55: return "🟠 Regular"
    return "🔴 Requiere práctica"

def barra_color(s):
    if s >= 80: return "normal"
    if s >= 60: return "normal"
    return "normal"   # st.progress solo acepta "normal" en versiones recientes

# ============================================================
# 10. GENERADOR DE REPORTE 100% GRATUITO
# ============================================================
def mostrar_reporte(sim: dict, disciplina: str, forma: str,
                    angulo: str, seg_usr: float) -> None:

    total   = sim["total"]
    brazos  = sim["brazos"]
    piernas = sim["piernas"]
    torso   = sim["torso"]
    hombros = sim["hombros"]
    caderas = sim["caderas"]
    equilib = sim["equilibrio"]

    # ── Encabezado ────────────────────────────────────────
    st.subheader("📊 Reporte de Evaluación")
    st.markdown(
        f"**Disciplina:** {disciplina} &nbsp;|&nbsp; "
        f"**Forma:** {forma} &nbsp;|&nbsp; "
        f"**Vista:** {angulo}"
    )
    if seg_usr > SEGUNDOS_SKIP_INICIO:
        st.caption(
            f"ℹ️ El análisis comenzó automáticamente en el segundo {seg_usr} "
            f"por baja iluminación al inicio del video."
        )

    st.divider()

    # ── 4 métricas principales ────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, label, valor in [
        (c1, "Similitud Total",  total),
        (c2, "Brazos / Sup.",    brazos),
        (c3, "Piernas / Inf.",   piernas),
        (c4, "Torso / Postura",  torso),
    ]:
        with col:
            color = color_score(valor)
            st.markdown(f"""
            <div class="metric-card" style="border-top-color:{color}">
                <div class="metric-valor" style="color:{color}">{valor}%</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Desglose por zona (expanders) ─────────────────────
    st.markdown("#### 🔍 Análisis detallado por zona corporal")

    zonas_info = [
        (
            "💪 Hombros y parte superior",  hombros,
            "Alineación de hombros en golpes y bloqueos.",
            "Practica frente al espejo elevando y bajando los brazos lentamente. "
            "Ambos hombros deben estar al mismo nivel en cada técnica.",
            "No eleves los hombros hacia las orejas — mantenlos relajados y hacia abajo."
        ),
        (
            "🦾 Brazos y extremidades superiores", brazos,
            "Extensión, ángulo y velocidad de brazos en golpes y bloqueos.",
            "Repite cada técnica 20 veces lentamente frente al espejo, "
            "luego 20 veces a velocidad normal.",
            "Compara la posición de tus codos con el video del maestro — "
            "es donde más diferencias suelen aparecer."
        ),
        (
            "🦵 Piernas y extremidades inferiores", piernas,
            "Posición, altura y estabilidad de piernas en cada técnica.",
            "Practica las posiciones base (kibon seogi) 5 minutos diarios, "
            "manteniendo cada postura 30 segundos.",
            "Las rodillas deben apuntar en la misma dirección que los pies en cada posición."
        ),
        (
            "⚖️ Caderas y centro de gravedad", caderas,
            "Rotación y posicionamiento de caderas durante los movimientos.",
            "Practica la rotación de caderas aislada: manos en la cadera, "
            "gira despacio en cada dirección.",
            "Las caderas deben rotar antes que los brazos — "
            "la cadera genera la potencia del movimiento."
        ),
        (
            "🎯 Balance y control de manos", equilib,
            "Control y precisión final de manos y muñecas.",
            "Al finalizar cada técnica, congela la posición 2 segundos "
            "antes de continuar.",
            "Revisa la tensión del puño y la posición de la muñeca "
            "al final de cada golpe o bloqueo."
        ),
        (
            "🏋️ Torso y postura central", torso,
            "Alineación general del tronco durante la forma.",
            "Practica con la espalda contra la pared para desarrollar "
            "conciencia de tu postura vertical.",
            "Mantén el abdomen ligeramente contraído durante toda la forma."
        ),
    ]

    for titulo, score, descripcion, ejercicio, consejo in zonas_info:
        color = color_score(score)
        nivel = nivel_texto(score)
        with st.expander(f"{titulo} — {nivel} ({score}%)"):
            st.markdown(f"**¿Qué se analizó?** {descripcion}")
            st.progress(score / 100)
            if score < 55:
                st.error("🔴 **Área prioritaria de mejora**")
                st.markdown(f"**Ejercicio recomendado:** {ejercicio}")
                st.markdown(f"**Consejo del instructor:** {consejo}")
            elif score < 70:
                st.warning("🟠 **Área con oportunidad de mejora**")
                st.markdown(f"**Consejo del instructor:** {consejo}")
            elif score < 85:
                st.info("🟡 **Buen nivel — ajuste fino recomendado**")
                st.markdown(f"**Consejo:** {consejo}")
            else:
                st.success("✅ **Excelente ejecución en esta zona**")
                st.markdown("Mantén este nivel y enfócate en las zonas con menor puntaje.")

    st.divider()

    # ── Resumen general ───────────────────────────────────
    st.markdown("#### 📝 Resumen General")

    scores_zonas = {
        "Brazos": brazos, "Piernas": piernas, "Torso": torso,
        "Hombros": hombros, "Caderas": caderas, "Balance": equilib,
    }
    peores = sorted(scores_zonas.items(), key=lambda x: x[1])[:3]

    if total >= 85:
        st.success(
            f"🏆 **¡Ejecución sobresaliente!** Lograste {total}% de similitud con el maestro. "
            f"Estás muy cerca de dominar esta forma. "
            f"Sigue practicando para perfeccionar los detalles finales."
        )
    elif total >= 70:
        st.info(
            f"💪 **Buen nivel de ejecución** — {total}% de similitud. "
            f"La estructura general de la forma está bien establecida. "
            f"Concéntrate en mejorar: **{peores[0][0]}** ({peores[0][1]}%) "
            f"y **{peores[1][0]}** ({peores[1][1]}%)."
        )
    elif total >= 55:
        st.warning(
            f"📚 **Nivel intermedio** — {total}% de similitud. "
            f"Hay áreas importantes que trabajar. Prioriza: "
            f"**{peores[0][0]}** ({peores[0][1]}%), "
            f"**{peores[1][0]}** ({peores[1][1]}%) y "
            f"**{peores[2][0]}** ({peores[2][1]}%)."
        )
    else:
        st.error(
            f"🔁 **Se recomienda más práctica** — {total}% de similitud. "
            f"Revisa el video de referencia varias veces antes de grabar de nuevo. "
            f"Áreas clave: **{peores[0][0]}**, **{peores[1][0]}** y **{peores[2][0]}**."
        )

    st.divider()

    # ── Plan de práctica ──────────────────────────────────
    st.markdown("#### 📅 Plan de práctica sugerido")
    ca, cb, cc = st.columns(3)
    with ca:
        st.markdown("""
        <div class="plan-card">
        <div class="plan-titulo">📺 Días 1 – 2: Estudio</div>
        • Estudia el video de referencia<br>
        • Practica cada movimiento por separado<br>
        • Enfócate en las zonas marcadas en 🔴
        </div>
        """, unsafe_allow_html=True)
    with cb:
        st.markdown("""
        <div class="plan-card">
        <div class="plan-titulo">🪞 Días 3 – 4: Práctica</div>
        • Practica la forma completa lentamente<br>
        • Usa el espejo para verificar postura<br>
        • Repite las secciones más difíciles
        </div>
        """, unsafe_allow_html=True)
    with cc:
        st.markdown("""
        <div class="plan-card">
        <div class="plan-titulo">🎬 Día 5: Evaluación</div>
        • Ejecuta la forma a velocidad normal<br>
        • Graba un nuevo video<br>
        • Analiza tu progreso en el sistema
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.success("Sigue practicando — ¡cada entrenamiento te acerca más al nivel del maestro! 🥋")

# ============================================================
# 11. BARRA LATERAL
# ============================================================
with st.sidebar:
    st.markdown("### 🥋 Sistema Tri-Fuerza")
    st.markdown(
        f'<span class="user-email-gray">👤 {st.session_state.email}</span>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚪 Salir", type="secondary"):
            st.session_state.clear()
            st.rerun()
    with col2:
        if st.button("❓ Ayuda"):
            st.session_state.mostrar_ayuda = not st.session_state.mostrar_ayuda
            st.rerun()

    if st.session_state.mostrar_ayuda:
        st.info("""
        **Pasos para analizar:**
        1. Selecciona Disciplina y Forma
        2. Elige el ángulo de cámara
        3. En "Estudiar Referencia" observa al maestro
        4. En "Analizar / Comparar" sube tu video
        5. Presiona **ANALIZAR** y revisa tu reporte

        💡 Si al inicio del video hay poca
        luz, el sistema lo ajusta solo.
        """)

    st.divider()

    disciplina_principal = st.selectbox(
        "Disciplina:",
        ["Taekwondo", "Haidong Gumdo", "Hapkido", "Forma Musical", "Torneo"]
    )

    formas         = []
    forma_sel      = None
    serie          = None
    sub_disciplina = None

    if disciplina_principal == "Torneo":
        sub_disciplina = st.selectbox(
            "Disciplina de Torneo:", ["Taekwondo", "Haidong Gumdo", "Hapkido"]
        )
        disc_activa = sub_disciplina

    elif disciplina_principal == "Forma Musical":
        sub_disciplina = st.selectbox(
            "Disciplina Base:", ["Taekwondo", "Haidong Gumdo", "Hapkido"]
        )
        disc_activa = sub_disciplina
        st.info("ℹ️ El sistema identificará la forma automáticamente.")
        forma_sel = "Identificación Automática"

    else:
        disc_activa = disciplina_principal

    if forma_sel != "Identificación Automática":
        if disc_activa == "Taekwondo":
            serie = st.selectbox("Serie:", ["Kichos", "Palgwes", "Taegeuks", "Judanyas"])
            if serie == "Kichos":
                formas = [f"Kicho {i}" for i in range(1, 5)]
            elif serie == "Palgwes":
                formas = [f"Palgwe {i}" for i in range(1, 9)]
            elif serie == "Taegeuks":
                formas = [f"Taegeuk {i}" for i in range(1, 9)]
            else:
                formas = ["Koryo", "Keumgang", "Taebaek", "Pyongwon"]

        elif disc_activa == "Haidong Gumdo":
            serie  = st.selectbox("Serie:", ["Ssangsu", "Yedo", "Simsang"])
            formas = [f"{serie} {i}" for i in range(1, 6)]

        elif disc_activa == "Hapkido":
            serie  = st.selectbox("Serie:", ["Dan Bong", "Jang Bong", "Defensa"])
            formas = [f"{serie} {i}" for i in range(1, 6)]

        if formas:
            forma_sel = st.selectbox("Forma:", formas)

    st.divider()

    angulo_sel = st.radio(
        "📷 Ángulo de cámara:",
        ["Frente", "Lado"],
        captions=["Vista frontal", "Vista lateral"]
    )
    st.caption("📌 Vista desde arriba — próximamente")

    modo_uso = st.radio("⚙️ Acción:", ["Estudiar Referencia", "Analizar / Comparar"])

# ============================================================
# 12. CUERPO PRINCIPAL
# ============================================================
st.markdown(
    f'<div class="titulo-modulo">🥋 {disciplina_principal} — '
    f'{forma_sel or "Selecciona una forma"}</div>',
    unsafe_allow_html=True
)

drive_id            = obtener_drive_id(disc_activa, serie or "", forma_sel or "", angulo_sel)
video_ref_disponible = drive_id is not None

# ── MODO: ESTUDIAR REFERENCIA ──────────────────────────────
if modo_uso == "Estudiar Referencia":
    if video_ref_disponible:
        st.info(f"📹 Video de referencia: **{forma_sel}** — Vista: **{angulo_sel}**")
        st.iframe(f"https://drive.google.com/file/d/{drive_id}/preview", height=460)
        st.success("✅ Observa con atención cada movimiento del maestro antes de grabar tu video.")
    else:
        st.markdown(f"""
        <div class="video-no-disponible">
            📹 Video de referencia no disponible aún para:<br>
            <strong>{forma_sel} — Vista {angulo_sel}</strong><br><br>
            Próximamente se agregará al catálogo.
        </div>
        """, unsafe_allow_html=True)

# ── MODO: ANALIZAR / COMPARAR ─────────────────────────────
else:
    col_ref, col_user = st.columns(2)

    with col_ref:
        st.subheader("✅ Referencia del Maestro")
        if video_ref_disponible:
            st.iframe(f"https://drive.google.com/file/d/{drive_id}/preview", height=320)
            st.caption(f"📹 {forma_sel} — Vista {angulo_sel}")
        else:
            st.markdown(f"""
            <div class="video-no-disponible">
                📹 Video no disponible aún para:<br>
                <strong>{forma_sel} — Vista {angulo_sel}</strong>
            </div>
            """, unsafe_allow_html=True)

    with col_user:
        st.subheader("📤 Tu Ejecución")
        video_usr = st.file_uploader(
            "Sube tu video aquí",
            type=["mp4", "mov", "avi"],
            key="usr_video"
        )
        if video_usr:
            st.video(video_usr)

    st.divider()

    # ── Botón ANALIZAR ────────────────────────────────────
    st.markdown('<div class="btn-analizar">', unsafe_allow_html=True)
    boton_analizar = st.button("🔍 ANALIZAR MI EJECUCIÓN")
    st.markdown('</div>', unsafe_allow_html=True)

    if boton_analizar:
        if not video_ref_disponible:
            st.warning(
                f"⚠️ El video de referencia para **{forma_sel} — Vista {angulo_sel}** "
                f"aún no está disponible en el catálogo."
            )
        elif not video_usr:
            st.warning("⚠️ Sube tu video de ejecución para poder analizar.")
        else:
            with st.status("🔍 Analizando tu ejecución...", expanded=True) as status:

                # PASO 1 — Descargar referencia
                st.write("⬇️ Obteniendo video de referencia desde Google Drive...")
                ruta_ref = descargar_video_drive(drive_id)
                if not ruta_ref:
                    status.update(label="❌ Error al obtener referencia", state="error")
                    st.error(
                        "No se pudo descargar el video de referencia. "
                        "Verifica que el permiso en Drive sea 'Cualquiera con el enlace'."
                    )
                    st.stop()
                mb = round(os.path.getsize(ruta_ref) / 1_000_000, 1)
                st.write(f"   ✅ Referencia lista ({mb} MB).")

                # PASO 2 — Guardar video del alumno
                st.write("💾 Preparando tu video...")
                tmp_usr = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp_usr.write(video_usr.read())
                tmp_usr.flush()
                tmp_usr.close()
                st.write("   ✅ Video recibido correctamente.")

                # PASO 3 — Extraer keypoints
                st.write("🦴 Detectando postura del maestro (MediaPipe)...")
                kp_ref, seg_ref, fv_ref = extraer_keypoints_video(ruta_ref)
                st.write(f"   ✅ {fv_ref} posiciones detectadas en el video de referencia.")

                st.write("🦴 Detectando tu postura...")
                kp_usr, seg_usr, fv_usr = extraer_keypoints_video(tmp_usr.name)
                if seg_usr > SEGUNDOS_SKIP_INICIO:
                    st.write(
                        f"   ℹ️ Inicio ajustado al segundo {seg_usr} "
                        f"(iluminación baja detectada automáticamente)."
                    )
                else:
                    st.write(f"   ✅ {fv_usr} posiciones detectadas en tu video.")

                # Validar mínimo de frames
                if fv_ref < FRAMES_MINIMOS or fv_usr < FRAMES_MINIMOS:
                    status.update(label="❌ No se pudo analizar", state="error")
                    if fv_usr < FRAMES_MINIMOS:
                        st.error(
                            "No se detectó suficiente movimiento en tu video.\n\n"
                            "Asegúrate de que:\n"
                            "• Tu cuerpo completo sea visible en cámara\n"
                            "• Haya buena iluminación durante toda la forma\n"
                            "• El video dure al menos 20 segundos"
                        )
                    else:
                        st.error("❌ Error procesando el video de referencia.")
                    for f in [ruta_ref, tmp_usr.name]:
                        try: os.unlink(f)
                        except: pass
                    st.stop()

                # PASO 4 — Comparar movimientos
                st.write("📐 Comparando movimientos con FastDTW...")
                similitudes = calcular_similitud(kp_ref, kp_usr)
                st.write("   ✅ Análisis de similitud completado.")

                # Limpiar archivos temporales
                for f in [ruta_ref, tmp_usr.name]:
                    try: os.unlink(f)
                    except: pass

                status.update(label="✅ Análisis completado", state="complete")

            # ── Mostrar reporte completo ──────────────────
            st.divider()
            mostrar_reporte(
                similitudes, disc_activa, forma_sel, angulo_sel, seg_usr
            )

# ── Pie de página ─────────────────────────────────────────
st.caption("🔒 Sistema Tri-Fuerza Coreana · Análisis biomecánico con MediaPipe + FastDTW · 100% Gratuito")
