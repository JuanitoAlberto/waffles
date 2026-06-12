"""Diagnosticos de baseline y datos crudos: timestamps, residuales, RMS,
waveforms individuales y heatmap por modulo."""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from waffles.data_classes.WaveformSet import WaveformSet

from pmtcheck_config import ANALYSIS_LABEL, TICK_NS


def plot_timestamp_diffs(wfset, bins=None):
    """Histograma de daq_window_timestamp - timestamp para todas las waveforms."""
    diffvalues = [wf.daq_window_timestamp - wf.timestamp for wf in wfset.waveforms]
    dmax = np.max(diffvalues)
    dmin = np.min(diffvalues)
    print(dmax, dmin, (dmax - dmin) * TICK_NS * 1e-9)

    if bins is None:
        bins = np.linspace(-50, 50, 100)
    plt.hist(diffvalues, bins=bins)
    plt.xlabel("daq_window_timestamp - timestamp [ticks]")
    plt.show()


def mean_baseline_by_channel(wfset, analysis_label=ANALYSIS_LABEL):
    """Baseline media de cada canal, como {(endpoint, channel): media}."""
    baselines_by_channel = defaultdict(list)
    for wf in wfset.waveforms:
        baseline = wf.analyses[analysis_label].result["baseline"]
        baselines_by_channel[(wf.endpoint, wf.channel)].append(baseline)

    return {
        channel: np.mean(baselines)
        for channel, baselines in baselines_by_channel.items()
    }


def plot_baseline_residuals(wfset, endpoint=None, channel=None,
                            analysis_label=ANALYSIS_LABEL, bins=100):
    """Distribucion de baseline - media de canal.

    Sin argumentos usa todas las waveforms; con endpoint y channel solo
    las de ese canal.
    """
    channel_means = mean_baseline_by_channel(wfset, analysis_label)

    residuals = []
    for wf in wfset.waveforms:
        if endpoint is not None and wf.endpoint != endpoint:
            continue
        if channel is not None and wf.channel != channel:
            continue
        baseline = wf.analyses[analysis_label].result["baseline"]
        residuals.append(baseline - channel_means[(wf.endpoint, wf.channel)])

    plt.figure(figsize=(8, 5))
    plt.hist(np.array(residuals), bins=bins, histtype="step", linewidth=1.8)
    plt.xlabel("Baseline - media de canal [ADC]")
    plt.ylabel("Numero de waveforms")
    if endpoint is not None and channel is not None:
        plt.title(f"Baseline residuals endpoint {endpoint}, channel {channel}")
    else:
        plt.title("Distribucion de fluctuaciones de baseline")
    plt.grid(alpha=0.3)
    plt.show()


def plot_baseline_rms(wfset, baseline_window=slice(0, 65),
                      analysis_label=ANALYSIS_LABEL, x_range=(0, 100), bins=50):
    """Distribucion del RMS de baseline para todas las waveforms y canales."""
    baseline_rms_values = []
    for wf in wfset.waveforms:
        baseline = wf.analyses[analysis_label].result["baseline"]
        baseline_region = wf.adcs[baseline_window] - baseline
        baseline_rms_values.append(np.sqrt(np.mean(baseline_region**2)))

    plt.figure(figsize=(8, 5))
    plt.hist(np.array(baseline_rms_values), bins=bins, range=x_range,
             histtype="step", linewidth=1.8)
    plt.xlim(x_range)
    plt.xlabel("Baseline RMS por waveform [ADC]")
    plt.ylabel("Numero de waveforms")
    plt.title("Distribucion del RMS de baseline para todas las waveforms")
    plt.grid(alpha=0.3)
    plt.show()


def plot_single_waveform(wfset, endpoint, channel, index,
                         analysis_label=ANALYSIS_LABEL, subtract_baseline=True):
    """Plotea una waveform concreta de un canal, cruda o con baseline restada."""

    def select_channel(wf):
        return wf.endpoint == endpoint and wf.channel == channel

    wfset_channel = WaveformSet.from_filtered_WaveformSet(
        wfset, select_channel, show_progress=False
    )

    if len(wfset_channel.waveforms) == 0:
        raise ValueError(f"No waveforms found for endpoint {endpoint}, channel {channel}")

    if index >= len(wfset_channel.waveforms):
        raise IndexError(
            f"index={index} but only "
            f"{len(wfset_channel.waveforms)} waveforms are available"
        )

    wf = wfset_channel.waveforms[index]
    baseline = wf.analyses[analysis_label].result["baseline"]

    y_raw = wf.adcs
    y_plot = y_raw - baseline if subtract_baseline else y_raw
    times_ns = np.arange(len(y_plot)) * TICK_NS

    plt.figure(figsize=(10, 5))
    plt.plot(times_ns, y_plot, color="black", linewidth=1.2)

    if subtract_baseline:
        plt.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.8,
                    label="baseline subtracted")
        plt.ylabel("Amplitude - baseline [ADC]")
    else:
        plt.axhline(baseline, color="red", linestyle="--", linewidth=1, alpha=0.8,
                    label=f"baseline = {baseline:.2f} ADC")
        plt.ylabel("Amplitude [ADC]")

    plt.xlabel("Time [ns]")
    plt.title(
        f"Single waveform | endpoint {endpoint}, channel {channel} | "
        f"waveform {index}/{len(wfset_channel.waveforms)-1}"
    )
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.show()


def plot_heatmap(wfset, detectors=None, **overrides):
    """Heatmap de todas las waveforms por modulo (pesado: ejecutar bajo demanda).

    overrides se pasa a plot_detectors para cambiar cualquier parametro.
    """
    from waffles.np02_utils.AutoMap import ordered_modules_pmt
    from waffles.np02_utils.PlotUtils import plot_detectors

    args = dict(
        mode="heatmap",
        analysis_label=ANALYSIS_LABEL,
        adc_range_above_baseline=1000,
        adc_range_below_baseline=-250,
        adc_bins=400,
        time_bins=wfset.points_per_wf // 2,
        filtering=4,
        share_y_scale=True,
        share_x_scale=True,
        wfs_per_axes=5000,
        zlog=True,
        width=1600,
        height=1200,
    )
    args.update(overrides)

    if detectors is None:
        detectors = ordered_modules_pmt

    return plot_detectors(wfset, detectors, **args)
