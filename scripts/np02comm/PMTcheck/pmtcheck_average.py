"""Waveform promedio por canal y ajuste con deconvolucion.

Estrategia de promediado: restar baseline waveform a waveform, alinear
los pulsos al tick mediano del maximo del canal, promediar (nanmean) y
denoising TV1D. Despues se ajusta con DeconvFitter (modelo LArXe por
defecto, que incluye la componente intermedia td).
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.np02_utils.AutoMap import getModuleName
from waffles.utils.deconvolution.DeconvFitter import DeconvFitter
from waffles.utils.denoising.tv1ddenoise import Denoise

from pmtcheck_config import (
    ANALYSIS_LABEL,
    AVG_PEAK_SEARCH_WINDOW,
    ENDPOINT,
    EXCLUDED_CHANNELS,
    PMT_LAYOUT,
    TICK_NS,
    WLS_BY_CHANNEL,
)

FIT_PARAM_LABELS = {
    'A': 'Norm.',
    'fp': 'A_\\text{fast}',
    'fs': 'A_\\text{slow}',
    't1': '\\tau_\\text{fast}',
    't3': '\\tau_\\text{slow}',
    'td': '\\tau_\\text{inter}',  # td es la constante de tiempo intermedia en el modelo LArXe
    't0': 't_\\text{0}',
    'sigma': '\\sigma',
}


def shift_waveform(y, shift, fill_value=np.nan):
    """Desplaza una waveform en el eje temporal rellenando con fill_value."""
    y_shifted = np.full_like(y, fill_value, dtype=float)

    if shift > 0:
        y_shifted[shift:] = y[:-shift]
    elif shift < 0:
        y_shifted[:shift] = y[-shift:]
    else:
        y_shifted[:] = y

    return y_shifted


def average_aligned(wfset, analysis_label=ANALYSIS_LABEL,
                    search_window=AVG_PEAK_SEARCH_WINDOW, denoise_filter=2):
    """Promedia las waveforms de un canal tras restar baseline y alinear los picos."""
    corrected_wfs = []
    peak_ticks = []

    for wf in wfset.waveforms:
        baseline = wf.analyses[analysis_label].result["baseline"]
        y = wf.adcs - baseline
        corrected_wfs.append(y)
        peak_ticks.append(np.argmax(y[search_window]) + search_window.start)

    peak_ticks = np.array(peak_ticks)

    # Usar como referencia el tick mediano del maximo de este canal.
    reference_tick = int(np.median(peak_ticks))

    aligned_wfs = [
        shift_waveform(y, reference_tick - peak_tick)
        for y, peak_tick in zip(corrected_wfs, peak_ticks)
    ]

    wfm = np.nanmean(aligned_wfs, axis=0)
    return Denoise().apply_denoise(wfm, filter=denoise_filter)


def pmt_average_and_fit(
    wfset,
    oneexp=False,
    fixed_tau_fast_ns=6.0,      # Fijar tau_fast, t1, a 6 ns durante el ajuste. None = libre.
    fit_limits_ns=(None, 10000),
    scinttype='larxe',          # Modelo LArXe para incluir la componente intermedia td.
    denoise_filter=2,
    average_store=None,         # dict opcional donde guardar la promedio como (ep, ch): wfm
    xlim=(0, 6000),
):
    """Calcula la waveform promedio del canal, la ajusta y la dibuja en los ejes actuales.

    Devuelve el DeconvFitter con el resultado del ajuste (deconv.deconvolved
    contiene la waveform promedio).
    """
    ep = wfset.waveforms[0].endpoint
    ch = wfset.waveforms[0].channel

    wfm = average_aligned(wfset, denoise_filter=denoise_filter)

    if average_store is not None:
        average_store[(ep, ch)] = wfm

    deconv = DeconvFitter(scinttype=scinttype)
    deconv.deconvolved = wfm
    deconv.fit(
        oneexp=oneexp,
        fit_limits_ns=list(fit_limits_ns),
        fixed_tau_fast_ns=fixed_tau_fast_ns,
    )

    nticks = len(deconv.deconvolved)
    times = np.linspace(0, nticks * TICK_NS, nticks, endpoint=False)

    plt.plot(times, deconv.deconvolved, '-', lw=2, color='k', label='data')

    if hasattr(deconv, 'fit_results') and len(deconv.fit_results) > 0:
        plt.plot(deconv.times, deconv.model(deconv.times, *deconv.fit_results),
                 color='r', zorder=100, label='Fit Model')

        fit_info = [
            f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {deconv.m.fval:.2f} / {deconv.m.ndof:.0f} = {deconv.m.fmin.reduced_chi2:.2f}",
        ]

        param_lines = {}
        param_order = []
        for p, v, e in zip(deconv.m.parameters, deconv.m.values, deconv.m.errors):
            if p in FIT_PARAM_LABELS:
                param_lines[p] = f"${FIT_PARAM_LABELS[p]}$ = ${v:.3f} \\pm {e:.3f}$"
                param_order.append(p)

        # Invertir posicion de tau_slow (t3) y tau_inter (td) en el display.
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
    plt.xlim(*xlim)

    return deconv


def plot_pmt_layout(
    wfset,
    endpoint=ENDPOINT,
    layout=PMT_LAYOUT,
    wls_by_channel=WLS_BY_CHANNEL,
    excluded_channels=EXCLUDED_CHANNELS,
    figsize=(26, 40),
    savepath=None,
    average_store=None,
    **fit_kwargs,
):
    """Rejilla con la posicion fisica de los PMTs: waveform promedio + fit por canal.

    Los canales excluidos y los PMTs no operativos se marcan con color y
    motivo en lugar del plot. fit_kwargs se pasa a pmt_average_and_fit.
    """
    nrows = len(layout)
    ncols = max(len(row) for row in layout)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)

    for row_idx, row in enumerate(layout):
        for col_idx, item in enumerate(row):
            ax = axs[row_idx, col_idx]

            if item is None:
                ax.axis("off")
                continue

            pmt_label, channel = item

            if channel is None:
                ax.set_facecolor("#fff0f0")
                ax.text(
                    0.5, 0.5,
                    f"{pmt_label}\nNot working",
                    ha="center", va="center",
                    color="red", fontweight="bold",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            wls = wls_by_channel.get(channel, "?")

            if channel in excluded_channels:
                reason, bg_color, txt_color = excluded_channels[channel]
                ax.set_facecolor(bg_color)
                ax.text(
                    0.5, 0.5,
                    f"{pmt_label}  |  Ch {channel}\nWLS: {wls}\n{reason}",
                    ha="center", va="center",
                    color=txt_color, fontweight="bold", fontsize=11,
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            def select_this_channel(wf, endpoint=endpoint, channel=channel):
                return wf.endpoint == endpoint and wf.channel == channel

            wfset_ch = WaveformSet.from_filtered_WaveformSet(
                wfset, select_this_channel, show_progress=False
            )

            plt.sca(ax)

            if len(wfset_ch.waveforms) == 0:
                ax.text(
                    0.5, 0.5,
                    f"{pmt_label}\nCh {channel} | WLS: {wls}\nNo waveforms",
                    ha="center", va="center",
                    color="red", fontweight="bold",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                pmt_average_and_fit(wfset_ch, average_store=average_store, **fit_kwargs)
                ax.set_title(f"{pmt_label} | Ch {channel} | WLS: {wls}", fontsize=9)

    # Leyenda de canales excluidos, agrupados por motivo.
    reason_groups = {}
    for ch, (reason, bg_color, txt_color) in excluded_channels.items():
        reason_groups.setdefault((reason, bg_color, txt_color), []).append(ch)

    legend_patches = [
        mpatches.Patch(
            facecolor=bg_color, edgecolor=txt_color,
            label=f"{reason}  (ch {', '.join(str(c) for c in sorted(chs))})",
        )
        for (reason, bg_color, txt_color), chs in reason_groups.items()
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=len(legend_patches),
               fontsize=12, bbox_to_anchor=(0.5, 0.985), framealpha=0.95)

    fig.subplots_adjust(hspace=0.5, wspace=0.35, top=0.96, bottom=0.04, left=0.05, right=0.98)

    if savepath is not None:
        plt.savefig(savepath, dpi=150)
    plt.show()

    return fig, axs
