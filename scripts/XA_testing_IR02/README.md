# XA_testing_IR02

Análisis de las XA (X-ARAPUCA) del setup de testing del laboratorio **IR02**.
Datos tomados con **CoMPASS** (CAEN), digitizador **DT2730**, firmware DPP-PSD.
Cada XA se lee por dos canales (CH0 y CH1).

Siguiendo el patrón de `np02comm/PMTcheck`: la lógica vive en módulos `.py` y los
notebooks solo orquestan y pintan (con `%autoreload`, los cambios en los `.py` se
reflejan en caliente). Los parámetros van en `xa_config.py` (no código comentado).

## Datos

En EOS (montaje del CernBox): `/eos/user/j/jlorenla/DUNE_Project/Lab/Test_XA_11-06-2026`,
con 3 campañas (proyectos de CoMPASS):

| Etiqueta lógica | Carpeta CoMPASS | Contenido |
|---|---|---|
| `spe` | `CoMPASS256_ExternalTrigger` | Trigger externo, nivel SPE (LED/laser) |
| `darkcurrent` | `CoMPASS265_DarkCurrent` | Corriente oscura (self-trigger) |
| `highlight` | `CoMPASS265_ExternalTriggerMuchaLuz` | Trigger externo, mucha luz |

Cada run es `DAQ/<run_id>/` con `RAW/DataR_CH{0,1}@DT2730_*.BIN` (waveforms
crudas), `settings.xml` e `<run_id>_info.txt`. Formato binario validado en
`xa_compass_reader.py` (cabecera de 2 B + eventos `board/ch/timestamp/energy/
energyShort/flags/code/nsamples/samples`, 3000 muestras de 4 ns).

## Flujo de trabajo

1. **Catálogo** (mapea cada run a sus condiciones):
   ```bash
   python xa_catalog.py
   ```
   Genera `runs_catalog.csv`. Rellena a mano `source` (LED/laser), `ov_V` y
   `light_level` desde el logbook; el resto se deriva del `info.txt`/`settings.xml`
   y se preserva al regenerar.

2. **(Opcional) Cache de WaveformSet** para los análisis que usen waffles:
   ```bash
   python xa_preprocess.py --campaign spe --run 1
   python xa_preprocess.py --campaign spe --all
   ```
   Los análisis basados en numpy (`load_run_arrays`) leen los `.BIN` directamente
   y no necesitan este paso.

3. **Notebooks** (independientes; kernel `dbt (3.10.10)`):
   - `00_run_catalog.ipynb` — construye/inspecciona el catálogo de runs.
   - `01_gain_snr_spe.ipynb` — **obj. 1**: ganancia y S/N por SPE; LED vs laser.
   - `02_light_channel_compare.ipynb` — **obj. 2**: respuesta de CH0/CH1; LED vs laser.
   - `03_waveform_adc_hist.ipynb` — **obj. 3**: waveforms, persistencia e histograma
     de ADC; energía on-board de CoMPASS vs carga de la waveform.
   - `04_dark_rate_ov.ipynb` — **obj. 4**: dark count rate a nivel de SPE y vs OV.
   - `05_led_stability.ipynb` — **obj. 5**: estabilidad del LED inter/intra-run.

## Módulos

| Fichero | Contenido |
|---|---|
| `xa_config.py` | Constantes (DT2730, ventanas, gates, calibración, dark rate), rutas EOS, esquema del catálogo |
| `xa_compass_reader.py` | Parser del binario CoMPASS `.BIN` → arrays numpy y → `WaveformSet` de waffles |
| `xa_io.py` | `load_run_arrays()` (rápido, numpy) y `load_run()` (WaveformSet con cache) |
| `xa_preprocess.py` | CLI: convierte runs a cache HDF5 de waffles |
| `xa_catalog.py` | Genera/lee `runs_catalog.csv` desde `info.txt`/`settings.xml` |
| `xa_waveforms.py` | Procesado base por waveform: baseline, polaridad, amplitud, cargas (qshort/qlong) |
| `xa_charge.py` | Histograma de calibración SPE, picos PE, **ganancia** y **S/N** |
| `xa_darkrate.py` | Staircase tasa-vs-umbral (PE), **DCR**, lectura de los CSV MCS |
| `xa_stability.py` | Seguimiento de un observable inter-run e intra-run |
| `xa_viewers.py` | Waveforms, persistencia, histogramas de ADC, comparación de canales |

## Notas

- **Polaridad negativa**: la `signal` se define con polaridad corregida
  (`POLARITY * (adc − baseline)`), pulsos positivos. Baseline en `BASELINE_WINDOW`.
- Las ventanas y gates de `xa_config.py` son un punto de partida; ajústalos en los
  notebooks copiando los diccionarios (`cfg.CALIB`, etc.), sin editar los módulos.
- El OV no está en los ficheros de CoMPASS (lo fija la fuente de bias externa):
  debe rellenarse en el catálogo para el objetivo 4.
