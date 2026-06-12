# PMTcheck

Análisis de waveforms de los PMTs de NP02 (endpoint 110). La lógica vive en
módulos `.py` y los notebooks solo orquestan y pintan (con `%autoreload`, los
cambios en los `.py` se reflejan en caliente).

## Flujo de trabajo

1. **Generar el cache** (una vez por run, fuera del notebook):
   ```bash
   python pmt_preprocess.py --run 43363
   ```
   Carga desde EOS, ejecuta `runBasicWfAnaNP02` y guarda en `cache/`.

2. **Notebooks** (independientes entre sí; cada uno arranca del cache):
   - `01_baseline_diagnostics.ipynb` — timestamps, residuales de baseline, RMS,
     waveforms individuales, visor de anómalas, heatmap.
   - `02_quality_selection.ipynb` — configurar/aplicar los filtros de calidad
     (`wfset_quality`) y diagnosticar las razones de rechazo.
   - `03_charge_and_averages.ipynb` — Qtotal/Qfast/Qslow, clusters de forma de
     pulso, waveform promedio con fit de deconvolución y layout de PMTs.

## Módulos

| Fichero | Contenido |
|---|---|
| `pmtcheck_config.py` | Constantes: ventanas, cortes de calidad, geometría (layout, WLS, canales excluidos) |
| `pmtcheck_io.py` | `load_wfset(run, use_cache=True)` |
| `pmtcheck_quality.py` | Métricas por waveform, estadística robusta por canal, `build_quality_wfset()` |
| `pmtcheck_charge.py` | `compute_charges()` (única definición de Qfast/Qslow/Qtotal), plots y clusters |
| `pmtcheck_average.py` | Promedio alineado + denoise, `pmt_average_and_fit()`, `plot_pmt_layout()` |
| `pmtcheck_diagnostics.py` | Plots de baseline/timestamps/waveform individual, heatmap |
| `pmtcheck_viewers.py` | Visores interactivos ipywidgets (anómalas, por canal, por cluster) |

Notas:

- El antiguo "filtro adicional de amplitud" está integrado en los cortes de
  calidad como `sustained_amp_min` (ventana 75–80; canal 16: 78–83). Su corte
  superior (`< 7000` ADC) lo cubre `peak_amp_abs_max`.
- El filtro `adjust_offset` (que no hacía nada: devolvía siempre `True`) se ha
  eliminado; el wfset cargado se usa directamente.
- `PMT_wf_analysis_legacy.ipynb` es el notebook monolítico original, conservado
  como referencia hasta validar que los nuevos notebooks reproducen sus
  resultados. No mantenerlo en paralelo: una vez validado, eliminarlo.
