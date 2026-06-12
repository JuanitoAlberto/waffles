"""Filtros de calidad para preparar la waveform promedio.

Calcula metricas por waveform, estadisticas robustas por canal (mediana
y MAD escalado) y aplica los cortes de pmtcheck_config para construir un
WaveformSet limpio con build_quality_wfset().

El antiguo "filtro adicional de amplitud" del notebook esta integrado
como el corte sustained_amp_min (razon de rechazo: sustained_amp_low);
su corte superior (< 7000 ADC) ya lo cubre peak_amp_abs_max.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from waffles.data_classes.WaveformSet import WaveformSet

from pmtcheck_config import (
    AMP_CHECK_WINDOWS,
    ANALYSIS_LABEL,
    QUALITY_CUTS,
    QUALITY_WINDOWS,
)

# Metricas para las que se calculan mediana y escala robusta por canal.
METRICS_FOR_ROBUST_STATS = [
    "baseline",
    "baseline_rms",
    "pre_max",
    "pre_integral_pos",
    "peak_amp",
    "peak_tick",
    "charge",
]


def robust_center_scale(values):
    """Devuelve mediana y escala robusta tipo sigma usando MAD."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = 1.4826 * mad

    if scale == 0:
        scale = np.std(values)
    if scale == 0:
        scale = 1.0

    return median, scale


def compute_quality_metrics(wf, wf_index, windows, amp_windows, analysis_label=ANALYSIS_LABEL):
    """Calcula las metricas usadas por los filtros para una waveform."""
    if analysis_label not in wf.analyses:
        raise RuntimeError(
            f"Waveform {wf_index} no tiene analisis '{analysis_label}'. "
            "Ejecuta runBasicWfAnaNP02 antes de estos filtros."
        )

    baseline = wf.analyses[analysis_label].result["baseline"]
    y = np.asarray(wf.adcs, dtype=float) - baseline

    baseline_region = y[windows["baseline"]]
    pre_region = y[windows["pretrigger"]]
    signal_region = y[windows["signal"]]
    charge_region = y[windows["charge"]]
    amp_window = amp_windows.get(wf.channel, amp_windows["default"])

    peak_rel_idx = int(np.argmax(signal_region))
    peak_tick = windows["signal"].start + peak_rel_idx
    peak_amp = float(signal_region[peak_rel_idx])

    return {
        "wf_index": wf_index,
        "wf_id": id(wf),
        "endpoint": wf.endpoint,
        "channel": wf.channel,
        "channel_key": (wf.endpoint, wf.channel),
        "baseline": float(baseline),
        # RMS real de la region de baseline: sqrt(mean((ADC - baseline)^2)).
        "baseline_rms": float(np.sqrt(np.mean(baseline_region**2))),
        "pre_max": float(np.max(pre_region)),
        "pre_integral_pos": float(np.sum(np.clip(pre_region, 0, None))),
        "peak_tick": peak_tick,
        "peak_amp": peak_amp,
        "charge": float(np.sum(charge_region)),
        "raw_adc_min": float(np.min(y)),
        "raw_adc_max": float(np.max(y)),
        # Minimo en la ventana de amplitud sostenida (antiguo filtro adicional).
        "amp_window_min": float(np.min(y[amp_window])),
    }


def evaluate_quality_record(rec, stats, cuts):
    """Devuelve la lista de razones de rechazo para un registro de metricas."""
    reasons = []

    baseline_stat = stats["baseline"]
    if abs(rec["baseline"] - baseline_stat["median"]) > cuts["baseline_nmad"] * baseline_stat["scale"]:
        reasons.append("baseline_shift")

    rms_stat = stats["baseline_rms"]
    if rec["baseline_rms"] > rms_stat["median"] + cuts["pre_rms_nmad"] * rms_stat["scale"]:
        reasons.append("baseline_rms_high")

    pre_max_stat = stats["pre_max"]
    if rec["pre_max"] > pre_max_stat["median"] + cuts["pre_max_nmad"] * pre_max_stat["scale"]:
        reasons.append("pretrigger_peak")

    pre_integral_stat = stats["pre_integral_pos"]
    if rec["pre_integral_pos"] > pre_integral_stat["median"] + cuts["pre_integral_nmad"] * pre_integral_stat["scale"]:
        reasons.append("pretrigger_charge")

    peak_amp_stat = stats["peak_amp"]
    if abs(rec["peak_amp"] - peak_amp_stat["median"]) > cuts["peak_amp_nmad"] * peak_amp_stat["scale"]:
        reasons.append("peak_amp_outlier")

    if rec["peak_amp"] < cuts["peak_amp_abs_min"]:
        reasons.append("peak_amp_low")
    if rec["peak_amp"] > cuts["peak_amp_abs_max"]:
        reasons.append("peak_amp_high")

    peak_tick_median = stats["peak_tick"]["median"]
    if abs(rec["peak_tick"] - peak_tick_median) > cuts["peak_tick_abs"]:
        reasons.append("peak_time_shift")

    charge_stat = stats["charge"]
    if abs(rec["charge"] - charge_stat["median"]) > cuts["charge_nmad"] * charge_stat["scale"]:
        reasons.append("charge_outlier")

    if cuts["adc_abs_max"] is not None:
        if abs(rec["raw_adc_min"]) > cuts["adc_abs_max"] or abs(rec["raw_adc_max"]) > cuts["adc_abs_max"]:
            reasons.append("raw_adc_saturation")

    # Saturacion negativa: ADC crudo muy negativo indica desbordamiento del ADC.
    if cuts.get("adc_min_threshold") is not None:
        if rec["raw_adc_min"] < cuts["adc_min_threshold"]:
            reasons.append("adc_negative_saturation")

    # Amplitud sostenida en la ventana del pulso (antiguo filtro adicional).
    if cuts.get("sustained_amp_min") is not None:
        if rec["amp_window_min"] <= cuts["sustained_amp_min"]:
            reasons.append("sustained_amp_low")

    return reasons


@dataclass
class QualityResult:
    """Resultado completo del filtrado, para diagnosticos y depuracion."""
    records: list
    records_by_channel: dict
    stats_by_channel: dict
    reasons_by_wf_id: dict
    pass_by_wf_id: dict
    reason_counts: Counter
    channel_summary: dict
    cuts: dict
    windows: dict
    amp_windows: dict


def build_quality_wfset(
    wfset,
    cuts=None,
    windows=None,
    amp_windows=None,
    analysis_label=ANALYSIS_LABEL,
    show_progress=True,
    verbose=True,
):
    """Aplica todos los filtros de calidad y devuelve (wfset_quality, QualityResult).

    cuts/windows/amp_windows actualizan los valores por defecto de
    pmtcheck_config sin modificarlos globalmente.
    """
    cuts = {**QUALITY_CUTS, **(cuts or {})}
    windows = {**QUALITY_WINDOWS, **(windows or {})}
    amp_windows = {**AMP_CHECK_WINDOWS, **(amp_windows or {})}

    records = []
    records_by_channel = defaultdict(list)

    n_wfs = len(wfset.waveforms)
    if verbose:
        print(f"Calculando metricas de calidad sobre {n_wfs} waveforms...")

    for wf_index, wf in enumerate(wfset.waveforms):
        if verbose and wf_index > 0 and wf_index % 100000 == 0:
            print(f"  procesadas {wf_index}/{n_wfs} waveforms", flush=True)
        rec = compute_quality_metrics(wf, wf_index, windows, amp_windows, analysis_label)
        records.append(rec)
        records_by_channel[rec["channel_key"]].append(rec)

    stats_by_channel = {}
    for channel_key, channel_records in records_by_channel.items():
        stats = {"n": len(channel_records)}
        for metric in METRICS_FOR_ROBUST_STATS:
            median, scale = robust_center_scale([rec[metric] for rec in channel_records])
            stats[metric] = {"median": median, "scale": scale}
        stats_by_channel[channel_key] = stats

    reasons_by_wf_id = {}
    pass_by_wf_id = {}
    reason_counts = Counter()
    channel_summary = defaultdict(lambda: {"total": 0, "pass": 0, "reject": 0})

    for rec in records:
        reasons = evaluate_quality_record(rec, stats_by_channel[rec["channel_key"]], cuts)
        passed = len(reasons) == 0

        reasons_by_wf_id[rec["wf_id"]] = reasons
        pass_by_wf_id[rec["wf_id"]] = passed

        summary = channel_summary[rec["channel_key"]]
        summary["total"] += 1
        if passed:
            summary["pass"] += 1
        else:
            summary["reject"] += 1
            reason_counts.update(reasons)

    def select_quality(wf):
        return pass_by_wf_id.get(id(wf), False)

    wfset_quality = WaveformSet.from_filtered_WaveformSet(
        wfset, select_quality, show_progress=show_progress
    )

    if verbose:
        n_pass = len(wfset_quality.waveforms)
        n_reject = n_wfs - n_pass
        print(f"Waveforms totales: {n_wfs}")
        print(f"Waveforms que pasan filtros de calidad: {n_pass} ({100*n_pass/n_wfs:.2f}%)")
        print(f"Waveforms rechazadas: {n_reject} ({100*n_reject/n_wfs:.2f}%)")
        print("\nRazones de rechazo mas frecuentes:")
        for reason, count in reason_counts.most_common():
            print(f"  {reason:22s}: {count}")

    result = QualityResult(
        records=records,
        records_by_channel=dict(records_by_channel),
        stats_by_channel=stats_by_channel,
        reasons_by_wf_id=reasons_by_wf_id,
        pass_by_wf_id=pass_by_wf_id,
        reason_counts=reason_counts,
        channel_summary=dict(channel_summary),
        cuts=cuts,
        windows=windows,
        amp_windows=amp_windows,
    )
    return wfset_quality, result


def plot_quality_diagnostics(result, endpoint, channel, metric="peak_amp"):
    """Diagnosticos rapidos de los filtros de calidad para un canal.

    Metricas disponibles para `metric`:
      "baseline"          baseline calculada por runBasicWfAnaNP02
      "baseline_rms"      RMS en la ventana de baseline
      "pre_max"           maximo positivo antes de la senal
      "pre_integral_pos"  carga positiva acumulada antes de la senal
      "peak_tick"         tick del maximo en la ventana de senal
      "peak_amp"          amplitud maxima en la ventana de senal
      "charge"            integral en la ventana de carga
      "raw_adc_min"       minimo ADC - baseline de la waveform
      "raw_adc_max"       maximo ADC - baseline de la waveform
      "amp_window_min"    minimo en la ventana de amplitud sostenida
    """
    channel_key = (endpoint, channel)
    records_ch = result.records_by_channel.get(channel_key, [])
    if len(records_ch) == 0:
        raise ValueError(f"No hay registros para endpoint {endpoint}, channel {channel}")

    passed_ch = np.array([result.pass_by_wf_id[rec["wf_id"]] for rec in records_ch], dtype=bool)
    values_ch = np.array([rec[metric] for rec in records_ch], dtype=float)

    print(f"Canal {channel_key}: {len(records_ch)} waveforms")
    print(f"  aceptadas: {np.sum(passed_ch)}")
    print(f"  rechazadas: {np.sum(~passed_ch)}")

    reason_counts_ch = Counter()
    for rec in records_ch:
        reason_counts_ch.update(result.reasons_by_wf_id[rec["wf_id"]])

    print("Razones de rechazo en este canal:")
    for reason, count in reason_counts_ch.most_common():
        print(f"  {reason:22s}: {count}")

    fig, axs = plt.subplots(1, 2, figsize=(13, 4))

    if len(result.reason_counts) > 0:
        reasons, counts = zip(*result.reason_counts.most_common())
        axs[0].bar(range(len(reasons)), counts)
        axs[0].set_xticks(range(len(reasons)))
        axs[0].set_xticklabels(reasons, rotation=45, ha="right")
        axs[0].set_ylabel("Numero de rechazos")
        axs[0].set_title("Razones de rechazo globales")
    else:
        axs[0].text(0.5, 0.5, "Sin rechazos", ha="center", va="center")
        axs[0].set_axis_off()

    axs[1].hist(values_ch[passed_ch], bins=60, histtype="step", linewidth=1.8, label="aceptadas")
    axs[1].hist(values_ch[~passed_ch], bins=60, histtype="step", linewidth=1.8, label="rechazadas")
    axs[1].set_xlabel(metric)
    axs[1].set_ylabel("Numero de waveforms")
    axs[1].set_title(f"{metric} en endpoint {endpoint}, ch {channel}")
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
