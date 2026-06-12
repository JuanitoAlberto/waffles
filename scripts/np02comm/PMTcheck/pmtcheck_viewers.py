"""Visores para inspeccionar waveforms individuales.

Cada visor tiene dos modos:
- sin widgets: funciones/metodos .plot(...) que reciben el indice a mano.
  Funcionan siempre.
- con slider (ipywidgets): metodos/funciones *interact*. En algunas
  combinaciones de VS Code + ipykernel el widget no llega a renderizar y
  la celda se queda ejecutando; en ese caso interrumpe la celda y usa el
  modo sin widgets.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from pmtcheck_config import ANALYSIS_LABEL, TICK_NS


def robust_z(values):
    """Z-score robusto usando mediana y MAD, menos sensible a outliers que media/std."""
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = 1.4826 * mad

    if scale == 0:
        scale = np.std(values)
    if scale == 0:
        return np.zeros_like(values)

    return (values - median) / scale


class AnomalousBrowser:
    """Ordena las waveforms de un canal por anomaly score y las plotea una a una.

    El calculo se hace una sola vez en el constructor; despues .plot(rank)
    es inmediato (rank=0 es la mas anomala).
    """

    def __init__(
        self,
        wfset,
        endpoint,
        channel,
        analysis_label=ANALYSIS_LABEL,
        n_keep=10000,
        baseline_window=slice(0, 65),
        peak_window=slice(60, 100),
        integral_window=slice(60, 160),
    ):
        t0 = time.time()

        self.endpoint = endpoint
        self.channel = channel
        self.analysis_label = analysis_label

        self.waveforms = [
            wf for wf in wfset.waveforms
            if wf.endpoint == endpoint and wf.channel == channel
        ]

        if len(self.waveforms) == 0:
            raise ValueError(f"No waveforms found for endpoint {endpoint}, channel {channel}")

        # Calcular metricas simples solo para las waveforms seleccionadas.
        self.metrics = []
        for idx, wf in enumerate(self.waveforms):
            if analysis_label not in wf.analyses:
                raise RuntimeError(
                    f"Waveform idx={idx} no tiene analisis '{analysis_label}'. "
                    "Ejecuta runBasicWfAnaNP02 primero."
                )

            baseline = wf.analyses[analysis_label].result["baseline"]
            y = wf.adcs - baseline

            peak_rel_idx = np.argmax(y[peak_window])
            peak_tick = peak_window.start + peak_rel_idx

            self.metrics.append({
                "idx": idx,
                "baseline": baseline,
                "pre_rms": np.std(y[baseline_window]),
                "peak_tick": peak_tick,
                "peak_amp": y[peak_tick],
                "integral": np.sum(y[integral_window]),
            })

        # Score grande = waveform mas rara respecto al comportamiento tipico del canal.
        anomaly_scores = np.sqrt(
            robust_z([m["pre_rms"] for m in self.metrics]) ** 2
            + robust_z([m["peak_tick"] for m in self.metrics]) ** 2
            + robust_z([m["peak_amp"] for m in self.metrics]) ** 2
            + robust_z([m["integral"] for m in self.metrics]) ** 2
        )

        for m, score in zip(self.metrics, anomaly_scores):
            m["anomaly_score"] = score

        self.n_to_show = min(n_keep, len(self.waveforms))
        self.anomaly_order = np.argsort(anomaly_scores)[::-1][:self.n_to_show]

        print(f"Waveforms analizadas: {len(self.waveforms)}")
        print(f"Waveforms ordenadas por anomaly score: {self.n_to_show}")
        print(f"Preparacion terminada en {time.time() - t0:.2f} s")

    def plot(self, rank=0):
        """Plotea la waveform numero `rank` del ranking de anomalia (0 = la mas rara)."""
        actual_idx = self.anomaly_order[rank]
        wf = self.waveforms[actual_idx]
        m = self.metrics[actual_idx]

        baseline = wf.analyses[self.analysis_label].result["baseline"]
        y = wf.adcs - baseline
        times_ns = np.arange(len(y)) * TICK_NS

        plt.figure(figsize=(11, 5))
        plt.plot(times_ns, y, color="black", linewidth=1.0, alpha=0.85, label="Selected waveform")
        plt.axvline(m["peak_tick"] * TICK_NS, color="tab:red", linestyle="--",
                    linewidth=1.0, alpha=0.8, label="Peak tick")
        plt.axhline(0, color="gray", linestyle=":", linewidth=1.0)

        info_text = (
            f"anomaly rank = {rank} / {self.n_to_show - 1}\n"
            f"waveform index in scan = {actual_idx}\n"
            f"baseline = {m['baseline']:.2f} ADC\n"
            f"pre-RMS = {m['pre_rms']:.2f} ADC\n"
            f"peak = {m['peak_amp']:.2f} ADC at tick {m['peak_tick']}\n"
            f"integral = {m['integral']:.2f} ADC*tick\n"
            f"anomaly score = {m['anomaly_score']:.2f}"
        )

        plt.gca().text(
            0.02, 0.98, info_text,
            transform=plt.gca().transAxes,
            va="bottom", ha="right", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="lightgray"),
        )

        plt.xlabel("Time [ns]")
        plt.ylabel("Amplitude - baseline [ADC]")
        plt.title(f"All waveforms | Endpoint {self.endpoint}, channel {self.channel}")
        plt.legend(fontsize=8, loc="upper right")
        plt.grid(alpha=0.3)
        plt.show()

    def interact(self):
        """Version con slider. Si la celda se queda ejecutando, usa .plot(rank)."""
        from ipywidgets import IntSlider, interact

        return interact(
            self.plot,
            rank=IntSlider(min=0, max=self.n_to_show - 1, step=1, value=0,
                           description="rank", continuous_update=False),
        )


def rank_anomalous(wfset, endpoint, channel, **kwargs):
    """Construye un AnomalousBrowser para un canal. Ver AnomalousBrowser."""
    return AnomalousBrowser(wfset, endpoint, channel, **kwargs)


def plot_channel_waveform(wfset, endpoint, channel, index, analysis_label=ANALYSIS_LABEL):
    """Plotea la waveform `index` de un canal (modo sin widgets)."""
    selected_wfs = [
        wf for wf in wfset.waveforms
        if wf.endpoint == endpoint and wf.channel == channel
    ]
    if len(selected_wfs) == 0:
        raise ValueError(f"No waveforms found for endpoint {endpoint}, channel {channel}")
    if index >= len(selected_wfs):
        raise IndexError(f"index={index} but only {len(selected_wfs)} waveforms are available")

    wf = selected_wfs[index]
    baseline = wf.analyses[analysis_label].result["baseline"]
    y = np.asarray(wf.adcs) - baseline

    plt.figure(figsize=(10, 4))
    plt.plot(y)
    plt.xlabel("Tick")
    plt.ylabel("ADC - baseline")
    plt.title(f"Endpoint {endpoint} | Channel {channel} | WF {index}/{len(selected_wfs)-1}")
    plt.grid(True)
    plt.show()


def browse_channel_waveforms(wfset, endpoint=110, analysis_label=ANALYSIS_LABEL):
    """Visor con widgets de waveforms por canal. Si la celda se queda
    ejecutando, usa plot_channel_waveform(wfset, endpoint, channel, index)."""
    from ipywidgets import Dropdown, IntSlider, interact

    channels = sorted({
        wf.channel for wf in wfset.waveforms if wf.endpoint == endpoint
    })

    if len(channels) == 0:
        raise ValueError(f"No waveforms found for endpoint {endpoint}")

    @interact(channel=Dropdown(options=channels, description="Channel"))
    def browse_channel(channel):
        selected_wfs = [
            wf for wf in wfset.waveforms
            if wf.endpoint == endpoint and wf.channel == channel
        ]

        @interact(idx=IntSlider(min=0, max=max(0, len(selected_wfs) - 1),
                                step=1, value=0, description="WF"))
        def show_waveform(idx):
            plot_channel_waveform(wfset, endpoint, channel, idx, analysis_label)


def plot_cluster_event(clusters, cluster_name, event_index):
    """Plotea un evento de un cluster Qfast/Qslow (modo sin widgets).

    clusters es el dict devuelto por pmtcheck_charge.split_clusters.
    """
    records = clusters[cluster_name]
    if len(records) == 0:
        raise ValueError(f"No waveforms found in cluster '{cluster_name}'.")

    event_index = min(event_index, len(records) - 1)
    rec = records[event_index]
    y = rec["y"]
    time_ns = np.arange(len(y)) * TICK_NS

    plt.figure(figsize=(10, 4.5))
    plt.plot(time_ns, y, linewidth=1.2)
    plt.axhline(0, color="black", linewidth=1, alpha=0.5)
    plt.axvline(rec["peak_tick"] * TICK_NS, color="tab:red", linestyle="--",
                linewidth=1.2, label="Peak tick")
    plt.axvspan(
        rec["qfast_start_tick"] * TICK_NS,
        rec["qfast_stop_tick"] * TICK_NS,
        color="tab:orange", alpha=0.25, label="Qfast window",
    )
    plt.axvspan(
        rec["qfast_start_tick"] * TICK_NS,
        rec["qtotal_stop_tick"] * TICK_NS,
        color="tab:blue", alpha=0.08, label="Qslow window",
    )
    plt.xlabel("Time [ns]")
    plt.ylabel("Amplitude above baseline [ADC]")
    plt.title(
        f"{cluster_name} event {event_index}/{len(records)-1} | "
        f"Qfast={rec['qfast']:.1f}, Qslow={rec['qslow']:.1f}, "
        f"Qfast/Qslow={rec['qfast_over_qslow']:.3f}"
    )
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def browse_cluster_events(clusters):
    """Visor con widgets de eventos por cluster. Si la celda se queda
    ejecutando, usa plot_cluster_event(clusters, cluster_name, event_index)."""
    from ipywidgets import Dropdown, IntSlider, interact

    max_cluster_size = max(len(records) for records in clusters.values())

    return interact(
        lambda cluster_name, event_index: plot_cluster_event(clusters, cluster_name, event_index),
        cluster_name=Dropdown(options=list(clusters.keys()), description="Cluster"),
        event_index=IntSlider(value=0, min=0, max=max_cluster_size - 1, step=1,
                              description="Event"),
    )
