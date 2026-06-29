"""Configuracion compartida del analisis de las XA (X-ARAPUCA) del IR02.

Datos tomados con CoMPASS (CAEN) y digitizador DT2730, firmware DPP-PSD, en el
setup de testing del laboratorio IR02. Cada XA se lee por dos canales del
digitizador (CH0 y CH1).

Aqui viven SOLO constantes y rutas. Los notebooks copian los diccionarios y los
modifican localmente si quieren probar otros valores (parametros, no codigo
comentado). La logica vive en los modulos xa_*.py.
"""

from pathlib import Path

# --- Rutas -------------------------------------------------------------------

# Raiz de los datos en EOS (montaje del CernBox del usuario).
EOS_BASE = Path(
    "/eos/user/j/jlorenla/DUNE_Project/Lab/Test_XA_11-06-2026"
)

# Carpeta de cada campana de medida dentro de EOS_BASE. La clave es la etiqueta
# "logica" que usamos en el analisis; el valor es el nombre real de la carpeta
# (proyecto de CoMPASS) en EOS.
CAMPAIGNS = {
    "spe":        "CoMPASS256_ExternalTrigger",         # Trigger externo, nivel SPE (LED/laser).
    "darkcurrent": "CoMPASS265_DarkCurrent",            # Corriente oscura (self-trigger).
    "highlight":  "CoMPASS265_ExternalTriggerMuchaLuz",  # Trigger externo, mucha luz.
}

# Cache local (HDF5 con WaveformSet ya convertido) y catalogo de runs.
CACHE_DIR = Path(__file__).parent / "cache"
CATALOG_CSV = Path(__file__).parent / "runs_catalog.csv"

# Subcarpeta donde CoMPASS guarda los .BIN crudos de cada run.
RAW_SUBDIR = "RAW"
# Subcarpeta con los CSV de MCS (rate vs tiempo) en las medidas de dark current.
OTHER_SUBDIR = "OTHER"

# --- Digitizador DT2730 (leido de settings.xml) ------------------------------

BOARD_MODEL = "DT2730"
ADC_BITS = 14                       # 0 .. 16383 cuentas.
ADC_MAX = (1 << ADC_BITS) - 1       # 16383
ADC_DYNAMIC_VPP = 2.0               # Rango dinamico de entrada (V) tipico del DT2730.

RAW_SAMPLE_NS = 2.0                 # Periodo de muestreo nativo: sampleTime=2000 ps.
WAVE_DOWNSAMPLING = 2               # SRV_PARAM_WAVEFORM_DOWNSAMPLING = WAVE_DOWNSAMPLING_X2.
SAMPLE_NS = RAW_SAMPLE_NS * WAVE_DOWNSAMPLING  # 4.0 ns por muestra de la waveform guardada.

POLARITY = -1                       # SRV_PARAM_CH_POLARITY = POLARITY_NEGATIVE (pulsos hacia abajo).
TIMESTAMP_UNIT_NS = 1e-3            # Timestamps de CoMPASS en ps -> ns.

# --- Identificacion de los canales de la XA ----------------------------------

# Endpoint ficticio para encajar en el modelo de waffles (Waveform.endpoint).
ENDPOINT = 2730
CHANNELS = [0, 1]

# Etiqueta fisica de cada canal de la XA. Rellena el recubrimiento/zona real
# cuando lo tengas del logbook del IR02.
CHANNEL_LABELS = {
    0: "CH0",
    1: "CH1",
}

# --- Ventanas y gates de integracion (en muestras de 4 ns) -------------------
# La waveform tiene 3000 muestras (12 us). Pretrigger ~ 600 muestras (2400 ns).

NSAMPLES = 3000                     # Record length (SRV_PARAM_RECORD_LENGTH efectivo).
PRETRIGGER_SAMPLES = 600            # SRV_PARAM_CH_PRETRG = 2400 ns / 4 ns.

# Ventana de baseline: zona plana antes del pico. Se deja margen respecto al pico.
BASELINE_WINDOW = slice(0, 500)

# Ventana de busqueda del pico (alrededor del pretrigger).
PEAK_SEARCH_WINDOW = slice(540, 760)

# Gates de carga, medidos desde el inicio del pico (en muestras).
# Q_short: integral rapida (SPE / componente rapida). Q_long: integral total.
QSHORT_GATE_SAMPLES = 25            # ~100 ns.
QLONG_GATE_SAMPLES = 250           # ~1 us.

# --- Histograma de calibracion SPE (objetivo 1 y 4) --------------------------

CALIB = {
    "observable": "qshort",        # "qshort" | "qlong" | "amplitude".
    "bins": 300,
    "range": None,                 # None = automatico (percentiles); o (min, max).
    "max_peaks": 5,                # Numero maximo de picos PE a buscar.
    "prominence": 0.02,            # Prominencia minima (fraccion del maximo) para find_peaks.
    "min_peak_distance_frac": 0.01,  # Distancia minima entre picos (fraccion del rango).
    "smooth_sigma_bins": 1.5,      # Suavizado gaussiano del histograma antes de buscar picos.
}

# --- Dark count rate / SPE a varios OV (objetivo 4) --------------------------

DARKRATE = {
    "threshold_pe": 0.5,           # Umbral en unidades de PE (a partir de la ganancia SPE).
    "amp_window": PEAK_SEARCH_WINDOW,  # Ventana donde se busca el pulso para contar tasas.
    "deadtime_samples": 50,        # Muestras de veto tras un pulso para no contar el mismo dos veces.
}

# Over-voltages medidos (V). Rellena con los valores reales del logbook; el
# catalogo de runs mapea cada run a su OV.
OV_VALUES = []

# --- Estabilidad del LED (objetivo 5) ----------------------------------------

STABILITY = {
    "observable": "qshort",        # Magnitud cuya media/anchura se sigue en el tiempo.
    "reference_channel": 0,        # Canal de referencia para seguir el LED.
}

# --- Catalogo de runs --------------------------------------------------------
# Columnas del CSV que mapea cada run a sus condiciones. Las columnas derivables
# del info.txt/settings.xml las rellena xa_catalog.py; el resto (source, ov_V,
# light_level) las rellena el usuario desde el logbook.

CATALOG_COLUMNS = [
    "campaign",        # spe | darkcurrent | highlight
    "run",             # numero de run (entero, unico dentro de la campana)
    "run_id",          # nombre de carpeta (p.ej. 11Junio_run_1)
    "source",          # LED | laser | none   (objetivos 1, 2)
    "ov_V",            # over-voltage en voltios (objetivo 4)
    "light_level",     # spe | high | dark | config
    "start_time",      # del info.txt
    "stop_time",       # del info.txt
    "duration_s",      # derivado
    "n_events_ch0",    # output counts CH0
    "n_events_ch1",    # output counts CH1
    "rate_ch0_cps",    # average rate CH0
    "rate_ch1_cps",    # average rate CH1
    "dc_offset_pct",   # SRV_PARAM_CH_BLINE_DCOFFSET
    "notes",
]
