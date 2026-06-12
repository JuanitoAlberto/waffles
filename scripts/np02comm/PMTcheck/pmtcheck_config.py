"""Configuracion compartida del analisis PMTcheck.

Constantes de adquisicion, ventanas y cortes de calidad, y datos de
geometria del detector (layout de PMTs, WLS y canales excluidos).
Los notebooks copian estos diccionarios y los modifican localmente si
quieren probar otros valores.
"""

from pathlib import Path

# --- Adquisicion y cache -----------------------------------------------------

TICK_NS = 16.0            # Periodo de muestreo de los PMTs: 1 tick = 16 ns.
ANALYSIS_LABEL = "std"    # Etiqueta del analisis creado por runBasicWfAnaNP02.

DEFAULT_DATADIR = "/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/"
CACHE_DIR = Path(__file__).parent / "cache"

# Parametros de runBasicWfAnaNP02 (usados por pmt_preprocess.py y load_wfset).
BASELINE_FINISH = 65
THRESHOLD = 25

# --- Ventanas para los filtros de calidad (en ticks) -------------------------

QUALITY_WINDOWS = {
    "baseline": slice(0, 65),      # Zona donde esperamos solo baseline.
    "pretrigger": slice(0, 55),    # Zona usada para buscar pulsos previos al trigger.
    "signal": slice(60, 110),      # Zona donde esperamos el pulso principal.
    "charge": slice(60, 180),      # Zona usada para estimar carga/integral del pulso.
}

# Ventana donde se exige amplitud sostenida por encima de sustained_amp_min
# (antiguo "filtro adicional de amplitud" del notebook). El canal 16 tiene el
# pulso desplazado y usa su propia ventana.
AMP_CHECK_WINDOWS = {
    "default": slice(75, 80),
    16: slice(78, 83),
}

# --- Cortes de calidad --------------------------------------------------------
# Los cortes *_nmad usan estadistica robusta por canal: mediana +/- N * MAD escalado.

QUALITY_CUTS = {
    "baseline_nmad": 6.0,          # Baseline muy desplazada respecto a la mediana del canal.
    "pre_rms_nmad": 5.0,           # Ruido alto en la ventana de baseline.
    "pre_max_nmad": 6.0,           # Pico positivo anomalo antes de la senal.
    "pre_integral_nmad": 6.0,      # Mucha carga positiva antes de la senal.
    "peak_amp_nmad": 6.0,          # Amplitud principal anomala respecto al canal.
    "charge_nmad": 6.0,            # Integral/carga anomala respecto al canal.
    "peak_tick_abs": 4,            # Pico principal demasiado desplazado, en ticks.
    "peak_amp_abs_min": 400.0,     # Corte absoluto minimo de amplitud de pico.
    "peak_amp_abs_max": 7000.0,    # Corte absoluto para saturados/eventos enormes.
    "adc_abs_max": None,           # Pon un valor si conoces el limite de saturacion ADC crudo.
    "adc_min_threshold": -1000.0,  # Rechaza eventos con algun ADC crudo < este valor (saturacion negativa).
    "sustained_amp_min": 400.0,    # min(y[AMP_CHECK_WINDOW]) debe superar este valor (pulso presente).
}

# --- Cargas (Qfast / Qslow / Qtotal) ------------------------------------------

PEAK_SEARCH_WINDOW = slice(60, 110)  # Ventana de busqueda del pico para las integrales.
QFAST_NS = 64.0                      # Integral rapida alrededor del pico.
QTOTAL_NS = 1300.0                   # Integral total/lenta desde el inicio del pico.

# --- Waveform promedio ----------------------------------------------------------

AVG_PEAK_SEARCH_WINDOW = slice(60, 100)  # Ventana del pico para alinear antes de promediar.

# --- Geometria del detector (endpoint 110) --------------------------------------

ENDPOINT = 110

# Lista de canales a plotear (orden descendente y de izq a dch).
CHANNELS_TO_ANALYZE = [14, 12, 37, 0, 17, 10, 16, 2, 7, 40, 6, 36, 34, 46, 24, 32, 42, 26, 22, 44, 30, 20]

# (etiqueta PMT, canal). None = hueco en la rejilla; canal None = PMT no operativo.
PMT_LAYOUT = [
    [None,             ("PMT 15", 14),   ("PMT 41", 12),  None],
    [("PMT 12", 37),   ("PMT 5", 0),     ("PMT 38", 17),  ("PMT 31", 10)],
    [("PMT 40", None), ("PMT 13", None), ("PMT 35", 16),  ("PMT 37", 2)],
    [("PMT 29", 7),    ("PMT 34", 40),   ("PMT 7", 6),    ("PMT 28", 36)],
    [("PMT 32", 34),   ("PMT 17", 46),   ("PMT 25", 24),  ("PMT 26", 32)],
    [("PMT 16", 42),   ("PMT 6", 26),    ("PMT 14", 22),  ("PMT 19", 44)],
    [None,             ("PMT 20", 30),   ("PMT 21", 20),  None],
]

WLS_BY_CHANNEL = {
    14: "PEN",   12: "PEN",   37: "TPB",   0: "TPB",
    17: "PEN",   10: "PEN",   16: "PEN+Q", 2: "PEN",
    7: "PEN",    40: "PEN",   6: "PEN",    36: "PEN",
    34: "PEN",   46: "PEN",   24: "PEN",   32: "PEN",
    42: "TPB",   26: "TPB",   22: "TPB",   44: "TPB",
    30: "PEN",   20: "CLEAR",
}

# canal: (motivo, color de fondo, color de texto)
EXCLUDED_CHANNELS = {
    10: ("Undershoot", "#FFF3CD", "#7D4E00"),
    14: ("Undershoot", "#FFF3CD", "#7D4E00"),
    36: ("Undershoot", "#FFF3CD", "#7D4E00"),
    0:  ("Saturation", "#CCE5FF", "#003D80"),
    6:  ("Saturation", "#CCE5FF", "#003D80"),
    30: ("Saturation", "#CCE5FF", "#003D80"),
    32: ("Saturation", "#CCE5FF", "#003D80"),
    42: ("Bad signal", "#F8D7DA", "#721C24"),
    44: ("Bad signal", "#F8D7DA", "#721C24"),
}
