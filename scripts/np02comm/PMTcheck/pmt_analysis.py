"""
PMT pulse-shape and average-waveform analysis helpers.

Functions
---------
compute_qtotal_amplitude      : Qtotal vs amplitude for one channel.
compute_qfast_qslow           : Qfast/Qslow PSD for one channel.
compute_pulse_shape_clusters  : Cluster waveforms by Qfast/Qslow region.
pmt_average_and_fit           : Aligned average waveform + deconvolution fit (plots inline).
plot_pmt_layout               : Full PMT grid with average waveforms.

Constants
---------
PMT_LAYOUT, WLS_BY_CHANNEL, EXCLUDED_CHANNELS : NP02 PMT geometry.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.utils.denoising.tv1ddenoise import Denoise
from waffles.utils.deconvolution.DeconvFitter import DeconvFitter
from waffles.np02_utils.AutoMap import getModuleName


TICK_NS = 16.0  # ns per ADC tick for PMTs

# ---------------------------------------------------------------------------
# NP02 PMT geometry constants
# ---------------------------------------------------------------------------

PMT_LAYOUT = [
    [None,            ("PMT 15", 14), ("PMT 41", 12), None           ],
    [("PMT 12", 37), ("PMT 5",   0), ("PMT 38", 17), ("PMT 31", 10) ],
    [("PMT 40", None),("PMT 13", None),("PMT 35", 16), ("PMT 37",  2)],
    [("PMT 29",  7), ("PMT 34", 40), ("PMT 7",   6), ("PMT 28", 36) ],
    [("PMT 32", 34), ("PMT 17", 46), ("PMT 25", 24), ("PMT 26", 32) ],
    [("PMT 16", 42), ("PMT 6",  26), ("PMT 14", 22), ("PMT 19", 44) ],
    [None,            ("PMT 20", 30), ("PMT 21", 20), None           ],
]

WLS_BY_CHANNEL = {
    14: "PEN",   12: "PEN",   37: "TPB",    0: "TPB",
    17: "PEN",   10: "PEN",   16: "PEN+Q",  2: "PEN",
     7: "PEN",   40: "PEN",    6: "PEN",   36: "PEN",
    34: "PEN",   46: "PEN",   24: "PEN",   32: "PEN",
    42: "TPB",   26: "TPB",   22: "TPB",   44: "TPB",
    30: "PEN",   20: "CLEAR",
}

# (label, background_color, text_color)
EXCLUDED_CHANNELS = {
    10: ("Undershoot", "#FFF3CD", "#7D4E00"),
    14: ("Undershoot", "#FFF3CD", "#7D4E00"),
    36: ("Undershoot", "#FFF3CD", "#7D4E00"),
     0: ("Saturation", "#CCE5FF", "#003D80"),
     6: ("Saturation", "#CCE5FF", "#003D80"),
    30: ("Saturation", "#CCE5FF", "#003D80"),
    32: ("Saturation", "#CCE5FF", "#003D80"),
    42: ("Bad signal", "#F8D7DA", "#721C24"),
    44: ("Bad signal", "#F8D7DA", "#721C24"),
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _baseline_subtract(wf, analysis_label="std"):
    baseline = wf.analyses[analysis_label].result["baseline"]
    return float(baseline), np.asarray(wf.adcs, dtype=float) - baseline


def _find_peak_tick(y, window):
    return window.start + int(np.argmax(y[window]))


# ---------------------------------------------------------------------------
# Charge analysis
# ---------------------------------------------------------------------------

def compute_qtotal_amplitude(
    wfset,
    endpoint,
    channel,
    peak_search_window=slice(60, 110),
    qtotal_duration_ns=1_300.0,
    analysis_label="std",
):
    """Compute (qtotal, amplitude) arrays for a single channel.

    Returns
    -------
    qtotal_vals : np.ndarray
    amplitude_vals : np.ndarray
    """
    qtotal_ticks  = int(np.round(qtotal_duration_ns / TICK_NS))
    qtotal_vals   = []
    amplitude_vals = []

    for wf in wfset.waveforms:
        if wf.endpoint != endpoint or wf.channel != channel:
            continue
        _, y       = _baseline_subtract(wf, analysis_label)
        peak_tick  = _find_peak_tick(y, peak_search_window)
        start      = max(peak_tick - 1, 0)
        stop       = min(start + qtotal_ticks, len(y))
        region     = y[start:stop]
        qtotal_vals.append(float(np.sum(region)))
        amplitude_vals.append(float(np.max(region)))

    if not qtotal_vals:
        raise ValueError(f"No waveforms found for endpoint={endpoint}, channel={channel}")
    return np.array(qtotal_vals), np.array(amplitude_vals)


def compute_qfast_qslow(
    wfset,
    endpoint,
    channel,
    peak_search_window=slice(60, 110),
    qfast_duration_ns=64.0,
    qtotal_duration_ns=1_300.0,
    analysis_label="std",
):
    """Compute (qfast, qslow) arrays for a single channel.

    Returns
    -------
    qfast_vals : np.ndarray
    qslow_vals : np.ndarray
    """
    qfast_ticks  = int(np.round(qfast_duration_ns  / TICK_NS))
    qtotal_ticks = int(np.round(qtotal_duration_ns / TICK_NS))
    qfast_vals   = []
    qslow_vals   = []

    for wf in wfset.waveforms:
        if wf.endpoint != endpoint or wf.channel != channel:
            continue
        _, y        = _baseline_subtract(wf, analysis_label)
        peak_tick   = _find_peak_tick(y, peak_search_window)
        start       = max(peak_tick - 1, 0)
        qfast_stop  = min(start + qfast_ticks,  len(y))
        qtotal_stop = min(start + qtotal_ticks, len(y))
        qfast_vals.append(float(np.sum(y[start:qfast_stop])))
        qslow_vals.append(float(np.sum(y[start:qtotal_stop])))

    if not qfast_vals:
        raise ValueError(f"No waveforms found for endpoint={endpoint}, channel={channel}")
    return np.array(qfast_vals), np.array(qslow_vals)


def compute_pulse_shape_clusters(
    wfset,
    endpoint,
    channel,
    deviating_qslow_range,
    deviating_qfast_range,
    peak_search_window=slice(60, 110),
    qfast_duration_ns=64.0,
    qtotal_duration_ns=1_300.0,
    analysis_label="std",
):
    """Cluster waveforms into 'deviating' / 'rest' by their Qfast-Qslow region.

    Parameters
    ----------
    deviating_qslow_range : (float, float)
        Qslow interval defining the deviating cluster.
    deviating_qfast_range : (float, float)
        Qfast interval defining the deviating cluster.

    Returns
    -------
    records : list[dict]
    clusters : dict[str, list[dict]]   ("deviating" / "rest")
    corrected_wfs : dict[str, list[np.ndarray]]
    mean_wfs : dict[str, np.ndarray]
    """
    qfast_ticks  = int(np.round(qfast_duration_ns  / TICK_NS))
    qtotal_ticks = int(np.round(qtotal_duration_ns / TICK_NS))

    records       = []
    clusters      = {"deviating": [], "rest": []}
    corrected_wfs = {"deviating": [], "rest": []}

    for wf_index, wf in enumerate(wfset.waveforms):
        if wf.endpoint != endpoint or wf.channel != channel:
            continue
        baseline, y = _baseline_subtract(wf, analysis_label)
        peak_tick   = _find_peak_tick(y, peak_search_window)
        start       = max(peak_tick - 1, 0)
        qfast_stop  = min(start + qfast_ticks,  len(y))
        qtotal_stop = min(start + qtotal_ticks, len(y))
        qfast = float(np.sum(y[start:qfast_stop]))
        qslow = float(np.sum(y[start:qtotal_stop]))

        is_deviating = (
            deviating_qslow_range[0] <= qslow <= deviating_qslow_range[1]
            and deviating_qfast_range[0] <= qfast <= deviating_qfast_range[1]
        )
        cluster_name = "deviating" if is_deviating else "rest"

        rec = {
            "wf_index":         wf_index,
            "cluster":          cluster_name,
            "waveform":         wf,
            "baseline":         float(baseline),
            "peak_tick":        int(peak_tick),
            "qfast_start_tick": int(start),
            "qfast_stop_tick":  int(qfast_stop),
            "qtotal_stop_tick": int(qtotal_stop),
            "qfast":            qfast,
            "qslow":            qslow,
            "qfast_over_qslow": qfast / qslow if qslow != 0 else np.nan,
        }
        records.append(rec)
        clusters[cluster_name].append(rec)
        corrected_wfs[cluster_name].append(y)

    if not records:
        raise ValueError(f"No waveforms found for endpoint={endpoint}, channel={channel}")

    mean_wfs = {
        name: np.mean(np.vstack(wfs), axis=0)
        for name, wfs in corrected_wfs.items()
        if wfs
    }
    return records, clusters, corrected_wfs, mean_wfs


# ---------------------------------------------------------------------------
# Average waveform + deconvolution fit
# ---------------------------------------------------------------------------

def _shift_waveform(y, shift, fill_value=np.nan):
    y_out = np.full_like(y, fill_value, dtype=float)
    if shift > 0:
        y_out[shift:] = y[:-shift]
    elif shift < 0:
        y_out[:shift] = y[-shift:]
    else:
        y_out[:] = y
    return y_out


# Populated by pmt_average_and_fit; keyed by (endpoint, channel).
average_wvf = {}


def pmt_average_and_fit(wfset: WaveformSet):
    """Compute aligned average waveform and fit with deconvolution. Plots inline.

    Stores the averaged waveform in the module-level ``average_wvf`` dict.
    """
    denoiser = Denoise()
    deconv   = DeconvFitter(scinttype='larxe')

    ep = wfset.waveforms[0].endpoint
    ch = wfset.waveforms[0].channel

    search_window = slice(60, 100)
    corrected_wfs = []
    peak_ticks    = []

    for wf in wfset.waveforms:
        baseline = wf.analyses["std"].result["baseline"]
        y        = wf.adcs - baseline
        corrected_wfs.append(y)
        peak_ticks.append(np.argmax(y[search_window]) + search_window.start)

    peak_ticks     = np.array(peak_ticks)
    reference_tick = int(np.median(peak_ticks))
    aligned_wfs    = [_shift_waveform(y, reference_tick - t) for y, t in zip(corrected_wfs, peak_ticks)]

    wfm = np.nanmean(aligned_wfs, axis=0)
    wfm = denoiser.apply_denoise(wfm, filter=2)

    deconv.deconvolved    = wfm
    average_wvf[(ep, ch)] = wfm

    deconv.fit(
        oneexp=False,
        fit_limits_ns=[None, 10000],
        fixed_tau_fast_ns=6.0,
    )

    nticks = len(deconv.deconvolved)
    times  = np.linspace(0, nticks * TICK_NS, nticks, endpoint=False)

    plt.plot(times, deconv.deconvolved, '-', lw=2, color='k', label='data')

    if hasattr(deconv, 'fit_results') and len(deconv.fit_results) > 0:
        plt.plot(deconv.times, deconv.model(deconv.times, *deconv.fit_results),
                 color='r', zorder=100, label='Fit Model')

        mapnames = {
            'A':     'Norm.',
            'fp':    'A_\\text{fast}',
            'fs':    'A_\\text{slow}',
            't1':    '\\tau_\\text{fast}',
            't3':    '\\tau_\\text{slow}',
            'td':    '\\tau_\\text{inter}',
            't0':    't_\\text{0}',
            'sigma': '\\sigma',
        }

        fit_info   = [f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {deconv.m.fval:.2f} / {deconv.m.ndof:.0f}"
                      f" = {deconv.m.fmin.reduced_chi2:.2f}"]
        param_lines = {}
        param_order = []
        for p, v, e in zip(deconv.m.parameters, deconv.m.values, deconv.m.errors):
            if p in mapnames:
                param_lines[p] = f"${mapnames[p]}$ = ${v:.3f} \\pm {e:.3f}$"
                param_order.append(p)

        if 't3' in param_order and 'td' in param_order:
            i_slow, i_inter = param_order.index('t3'), param_order.index('td')
            param_order[i_slow], param_order[i_inter] = param_order[i_inter], param_order[i_slow]

        for p in param_order:
            fit_info.append(param_lines[p])

        plt.plot([], [], ' ', label='\n'.join(fit_info))

    plt.title(f"{getModuleName(ep, ch)}: {ep}-{ch}")
    plt.yscale('symlog', linthresh=10)
    plt.ylim(-10, np.max(wfm) * 1.5)
    plt.legend(fontsize=8)
    plt.ylabel('Amplitude [ADC]')
    plt.xlabel('Time [ns]')
    plt.xlim(0, 6000)


# ---------------------------------------------------------------------------
# PMT layout grid
# ---------------------------------------------------------------------------

def plot_pmt_layout(
    wfset,
    endpoint=110,
    layout=None,
    wls_by_channel=None,
    excluded_channels=None,
    savefig=None,
):
    """Render the PMT layout grid with average waveforms and deconvolution fits.

    Parameters
    ----------
    wfset : WaveformSet
        Filtered waveform set (output of quality selection + amplitude filter).
    endpoint : int
    layout : list[list] or None
        Use PMT_LAYOUT by default.
    wls_by_channel : dict or None
        Use WLS_BY_CHANNEL by default.
    excluded_channels : dict or None
        Use EXCLUDED_CHANNELS by default.
    savefig : str or None
        If given, save the figure to this path.
    """
    if layout           is None: layout           = PMT_LAYOUT
    if wls_by_channel   is None: wls_by_channel   = WLS_BY_CHANNEL
    if excluded_channels is None: excluded_channels = EXCLUDED_CHANNELS

    nrows = len(layout)
    ncols = max(len(row) for row in layout)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(26, 40), squeeze=False)

    for row_idx, row in enumerate(layout):
        for col_idx, item in enumerate(row):
            ax = axs[row_idx, col_idx]

            if item is None:
                ax.axis("off")
                continue

            pmt_label, channel = item
            wls = wls_by_channel.get(channel, "?")

            if channel is None:
                ax.set_facecolor("#fff0f0")
                ax.text(0.5, 0.5, f"{pmt_label}\nNot working",
                        ha="center", va="center", color="red", fontweight="bold",
                        transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            if channel in excluded_channels:
                reason, bg_color, txt_color = excluded_channels[channel]
                ax.set_facecolor(bg_color)
                ax.text(0.5, 0.5, f"{pmt_label}  |  Ch {channel}\nWLS: {wls}\n{reason}",
                        ha="center", va="center", color=txt_color, fontweight="bold",
                        fontsize=11, transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            def _select(wf, ep=endpoint, ch=channel):
                return wf.endpoint == ep and wf.channel == ch

            wfset_ch = WaveformSet.from_filtered_WaveformSet(wfset, _select, show_progress=False)
            plt.sca(ax)

            if len(wfset_ch.waveforms) == 0:
                ax.text(0.5, 0.5, f"{pmt_label}\nCh {channel} | WLS: {wls}\nNo waveforms",
                        ha="center", va="center", color="red", fontweight="bold",
                        transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                pmt_average_and_fit(wfset_ch)
                ax.set_title(f"{pmt_label} | Ch {channel} | WLS: {wls}", fontsize=9)

    legend_patches = [
        mpatches.Patch(facecolor="#FFF3CD", edgecolor="#7D4E00", label="Undershoot  (ch 10, 14, 36)"),
        mpatches.Patch(facecolor="#CCE5FF", edgecolor="#003D80", label="Saturation  (ch 0, 6, 30, 32)"),
        mpatches.Patch(facecolor="#F8D7DA", edgecolor="#721C24", label="Bad signal  (ch 42, 44)"),
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=3, fontsize=12,
               bbox_to_anchor=(0.5, 0.985), framealpha=0.95)
    fig.subplots_adjust(hspace=0.5, wspace=0.35, top=0.96, bottom=0.04, left=0.05, right=0.98)

    if savefig:
        plt.savefig(savefig, dpi=150)
    plt.show()
