"""Analisis de carga por canal: Qtotal, Qfast/Qslow y clusters de forma de pulso.

Una unica funcion compute_charges() define las integrales para todos los
plots, en lugar de repetir el calculo en cada celda del notebook.
"""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from pmtcheck_config import (
    ANALYSIS_LABEL,
    PEAK_SEARCH_WINDOW,
    QFAST_NS,
    QTOTAL_NS,
    TICK_NS,
)

QFAST_TICKS = int(np.round(QFAST_NS / TICK_NS))
QTOTAL_TICKS = int(np.round(QTOTAL_NS / TICK_NS))

QSLOW_LABEL = f"Qslow, {QTOTAL_NS/1000:.1f} us integral from peak start [ADC ticks]"
QFAST_LABEL = f"Qfast, {QFAST_NS:.0f} ns integral around peak [ADC ticks]"


def compute_charges(y, peak_search_window=PEAK_SEARCH_WINDOW):
    """Calcula pico e integrales de carga para una waveform ya corregida de baseline."""
    peak_region = y[peak_search_window]
    if len(peak_region) == 0:
        raise ValueError("Peak search window is empty for this waveform.")

    peak_tick = peak_search_window.start + int(np.argmax(peak_region))
    qfast_start_tick = max(peak_tick - 1, 0)
    qfast_stop_tick = min(qfast_start_tick + QFAST_TICKS, len(y))
    qtotal_stop_tick = min(qfast_start_tick + QTOTAL_TICKS, len(y))

    qfast = float(np.sum(y[qfast_start_tick:qfast_stop_tick]))
    qslow = float(np.sum(y[qfast_start_tick:qtotal_stop_tick]))
    amplitude = float(np.max(y[qfast_start_tick:qtotal_stop_tick]))

    return {
        "peak_tick": peak_tick,
        "qfast_start_tick": qfast_start_tick,
        "qfast_stop_tick": qfast_stop_tick,
        "qtotal_stop_tick": qtotal_stop_tick,
        "qfast": qfast,
        "qslow": qslow,  # tambien llamado Qtotal en los plots
        "qfast_over_qslow": qfast / qslow if qslow != 0 else np.nan,
        "amplitude": amplitude,
    }


@dataclass
class ChannelCharges:
    """Registros de carga de un canal: un dict por waveform (ver compute_charges)."""
    endpoint: int
    channel: int
    records: list = field(default_factory=list)

    def values(self, key):
        return np.array([rec[key] for rec in self.records])


def channel_charges(wfset, endpoint, channel, analysis_label=ANALYSIS_LABEL,
                    peak_search_window=PEAK_SEARCH_WINDOW):
    """Calcula las cargas de todas las waveforms de un canal."""
    cc = ChannelCharges(endpoint=endpoint, channel=channel)

    for wf_index, wf in enumerate(wfset.waveforms):
        if wf.endpoint != endpoint or wf.channel != channel:
            continue

        if analysis_label not in wf.analyses:
            raise RuntimeError(
                f"Waveform has no analysis '{analysis_label}'. Run runBasicWfAnaNP02 first."
            )

        baseline = wf.analyses[analysis_label].result["baseline"]
        y = np.asarray(wf.adcs, dtype=float) - baseline

        rec = compute_charges(y, peak_search_window)
        rec.update({
            "wf_index": wf_index,
            "waveform": wf,
            "baseline": float(baseline),
            "y": y,
        })
        cc.records.append(rec)

    if len(cc.records) == 0:
        raise ValueError(f"No filtered waveforms found for endpoint {endpoint}, channel {channel}")

    return cc


def plot_qtotal_vs_amplitude(cc):
    """Qtotal versus amplitude para un canal."""
    plt.figure(figsize=(8, 6))
    plt.scatter(cc.values("amplitude"), cc.values("qslow"), s=2, alpha=0.15, rasterized=True)
    plt.xlabel("Amplitude above baseline [ADC]")
    plt.ylabel(QSLOW_LABEL.replace("Qslow", "Qtotal"))
    plt.title(f"Qtotal versus amplitude after filtering | endpoint {cc.endpoint}, channel {cc.channel}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_qfast_vs_qslow(cc):
    """Qfast versus Qslow para un canal."""
    plt.figure(figsize=(8, 6))
    plt.scatter(cc.values("qslow"), cc.values("qfast"), s=2, alpha=0.15, rasterized=True)
    plt.xlabel(QSLOW_LABEL)
    plt.ylabel(QFAST_LABEL)
    plt.title(f"Qfast versus Qslow after filtering | endpoint {cc.endpoint}, channel {cc.channel}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_ratio_vs_qslow(cc):
    """Qfast/Qslow versus Qslow para un canal."""
    qslow = cc.values("qslow")
    ratio = cc.values("qfast_over_qslow")
    valid = np.isfinite(ratio)

    plt.figure(figsize=(8, 6))
    plt.scatter(qslow[valid], ratio[valid], s=4, alpha=0.25, rasterized=True)
    plt.xlabel(QSLOW_LABEL)
    plt.ylabel("Qfast / Qslow")
    plt.title(f"Qfast/Qslow versus Qslow after filtering | endpoint {cc.endpoint}, channel {cc.channel}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def split_clusters(cc, qslow_range, qfast_range):
    """Divide los registros en clusters 'deviating' / 'rest' segun rangos de Qslow y Qfast.

    Anota rec["cluster"] en cada registro y devuelve {nombre: [registros]}.
    """
    clusters = {"deviating": [], "rest": []}

    for rec in cc.records:
        is_deviating = (
            qslow_range[0] <= rec["qslow"] <= qslow_range[1]
            and qfast_range[0] <= rec["qfast"] <= qfast_range[1]
        )
        rec["cluster"] = "deviating" if is_deviating else "rest"
        clusters[rec["cluster"]].append(rec)

    return clusters


def plot_cluster_averages(cc, clusters):
    """Waveform promedio de cada cluster Qfast/Qslow."""
    mean_wfs = {}
    for cluster_name, records in clusters.items():
        if len(records) == 0:
            raise ValueError(f"No waveforms found in cluster '{cluster_name}'. Check the Qfast/Qslow ranges.")
        mean_wfs[cluster_name] = np.mean(np.vstack([rec["y"] for rec in records]), axis=0)

    time_ns = np.arange(len(next(iter(mean_wfs.values())))) * TICK_NS

    plt.figure(figsize=(9, 5))
    for cluster_name, mean_wf in mean_wfs.items():
        plt.plot(time_ns, mean_wf, label=f"{cluster_name}, n={len(clusters[cluster_name])}", linewidth=2)
    plt.axhline(0, color="black", linewidth=1, alpha=0.5)
    plt.xlabel("Time [ns]")
    plt.ylabel("Amplitude above baseline [ADC]")
    plt.title(f"Average waveforms by Qfast/Qslow cluster | endpoint {cc.endpoint}, channel {cc.channel}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    for cluster_name, records in clusters.items():
        print(f"{cluster_name}: {len(records)} waveforms")

    return mean_wfs


def plot_cluster_scatter(cc, qslow_range):
    """Qfast/Qslow versus Qslow coloreado por cluster (requiere split_clusters)."""
    if "cluster" not in cc.records[0]:
        raise RuntimeError("Run split_clusters first.")

    qslow = cc.values("qslow")
    ratio = cc.values("qfast_over_qslow")
    names = np.array([rec["cluster"] for rec in cc.records])
    valid = np.isfinite(ratio)

    plt.figure(figsize=(8, 6))
    for cluster_name, color in [("deviating", "tab:orange"), ("rest", "tab:blue")]:
        mask = valid & (names == cluster_name)
        plt.scatter(
            qslow[mask], ratio[mask],
            s=4, alpha=0.25, color=color,
            label=f"{cluster_name}, n={np.sum(mask)}",
            rasterized=True,
        )

    plt.axvspan(qslow_range[0], qslow_range[1], color="tab:orange", alpha=0.08)
    plt.xlabel(QSLOW_LABEL)
    plt.ylabel("Qfast / Qslow")
    plt.title(f"Qfast/Qslow versus Qslow | endpoint {cc.endpoint}, channel {cc.channel}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
