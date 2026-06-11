"""
Quality filters for PMT waveform analysis.

Provides functions to compute per-waveform quality metrics and apply
channel-wise robust cuts to build a clean WaveformSet.

Typical usage from a notebook
------------------------------
    from quality_filter import DEFAULT_QUALITY_WINDOWS, DEFAULT_QUALITY_CUTS
    from quality_filter import build_wfset_quality

    my_cuts = {**DEFAULT_QUALITY_CUTS, "peak_amp_abs_min": 500.0}
    result = build_wfset_quality(wfset_triggered_all, cuts=my_cuts)
    wfset_quality, stats_by_channel, pass_by_wf_id, reasons_by_wf_id, reason_counts, channel_summary = result
"""

from collections import defaultdict, Counter
import numpy as np

from waffles.data_classes.WaveformSet import WaveformSet


DEFAULT_ANALYSIS_LABEL = "std"

DEFAULT_QUALITY_WINDOWS = {
    "baseline":   slice(0, 65),   # Region expected to contain only baseline.
    "pretrigger": slice(0, 55),   # Region scanned for pre-trigger pulses.
    "signal":     slice(60, 110), # Region where the main pulse is expected.
    "charge":     slice(60, 180), # Region used for charge/integral estimation.
}

DEFAULT_QUALITY_CUTS = {
    "baseline_nmad":     6.0,    # Baseline shifted far from channel median.
    "pre_rms_nmad":      5.0,    # High noise in baseline window.
    "pre_max_nmad":      6.0,    # Spurious positive peak before signal.
    "pre_integral_nmad": 6.0,    # Excess positive charge before trigger.
    "peak_amp_nmad":     6.0,    # Main pulse amplitude outlier.
    "charge_nmad":       6.0,    # Integrated charge outlier.
    "peak_tick_abs":     4,      # Main peak displaced by more than N ticks.
    "peak_amp_abs_min":  400.0,  # Minimum acceptable amplitude [ADC].
    "peak_amp_abs_max":  7000.0, # Maximum acceptable amplitude [ADC].
    "adc_abs_max":       None,   # If set, reject |raw ADC| > this value.
    "adc_min_threshold": -1000.0 # Reject waveforms with raw ADC < this (negative saturation).
}


# ---------------------------------------------------------------------------
# Low-level utilities
# ---------------------------------------------------------------------------

def robust_center_scale(values):
    """Return (median, robust sigma) computed via MAD scaling."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    median = np.median(values)
    mad    = np.median(np.abs(values - median))
    scale  = 1.4826 * mad
    if scale == 0:
        scale = float(np.std(values))
    if scale == 0:
        scale = 1.0
    return float(median), float(scale)


def compute_quality_metrics(wf, wf_index, windows=None, analysis_label=DEFAULT_ANALYSIS_LABEL):
    """Compute quality metrics for a single waveform.

    Parameters
    ----------
    wf : Waveform
    wf_index : int
        Position index used only for error messages.
    windows : dict[str, slice], optional
        Override DEFAULT_QUALITY_WINDOWS.
    analysis_label : str
        Key of the analysis result that contains 'baseline'.

    Returns
    -------
    dict with keys: wf_index, wf_id, endpoint, channel, channel_key,
        baseline, baseline_rms, pre_max, pre_integral_pos, peak_tick,
        peak_amp, charge, raw_adc_min, raw_adc_max.
    """
    if windows is None:
        windows = DEFAULT_QUALITY_WINDOWS

    if analysis_label not in wf.analyses:
        raise RuntimeError(
            f"Waveform {wf_index} has no analysis '{analysis_label}'. "
            "Run runBasicWfAnaNP02 first."
        )

    baseline = wf.analyses[analysis_label].result["baseline"]
    y        = np.asarray(wf.adcs, dtype=float) - baseline

    baseline_region = y[windows["baseline"]]
    pre_region      = y[windows["pretrigger"]]
    signal_region   = y[windows["signal"]]
    charge_region   = y[windows["charge"]]

    peak_rel_idx = int(np.argmax(signal_region))
    peak_tick    = windows["signal"].start + peak_rel_idx

    return {
        "wf_index":         wf_index,
        "wf_id":            id(wf),
        "endpoint":         wf.endpoint,
        "channel":          wf.channel,
        "channel_key":      (wf.endpoint, wf.channel),
        "baseline":         float(baseline),
        "baseline_rms":     float(np.sqrt(np.mean(baseline_region ** 2))),
        "pre_max":          float(np.max(pre_region)),
        "pre_integral_pos": float(np.sum(np.clip(pre_region, 0, None))),
        "peak_tick":        peak_tick,
        "peak_amp":         float(signal_region[peak_rel_idx]),
        "charge":           float(np.sum(charge_region)),
        "raw_adc_min":      float(np.min(y)),
        "raw_adc_max":      float(np.max(y)),
    }


_STAT_METRICS = [
    "baseline", "baseline_rms", "pre_max",
    "pre_integral_pos", "peak_amp", "peak_tick", "charge",
]

def compute_stats_by_channel(records_by_channel):
    """Compute robust (median, scale) per metric, per channel.

    Parameters
    ----------
    records_by_channel : dict[(ep, ch), list[dict]]

    Returns
    -------
    dict[(ep, ch), dict]
        Each value has keys 'n' and one key per metric in _STAT_METRICS,
        each containing {'median': float, 'scale': float}.
    """
    stats = {}
    for channel_key, records in records_by_channel.items():
        ch_stats = {"n": len(records)}
        for metric in _STAT_METRICS:
            values = [r[metric] for r in records]
            median, scale = robust_center_scale(values)
            ch_stats[metric] = {"median": median, "scale": scale}
        stats[channel_key] = ch_stats
    return stats


def evaluate_quality_record(rec, stats_by_channel, cuts):
    """Return list of rejection reason strings for one quality record."""
    stats   = stats_by_channel[rec["channel_key"]]
    reasons = []

    def _nmad_shifted(metric, cut_key):
        stat = stats[metric]
        return abs(rec[metric] - stat["median"]) > cuts[cut_key] * stat["scale"]

    def _nmad_high(metric, cut_key):
        stat = stats[metric]
        return rec[metric] > stat["median"] + cuts[cut_key] * stat["scale"]

    if _nmad_shifted("baseline",   "baseline_nmad"):     reasons.append("baseline_shift")
    if _nmad_high("baseline_rms",  "pre_rms_nmad"):      reasons.append("baseline_rms_high")
    if _nmad_high("pre_max",       "pre_max_nmad"):      reasons.append("pretrigger_peak")
    if _nmad_high("pre_integral_pos", "pre_integral_nmad"): reasons.append("pretrigger_charge")
    if _nmad_shifted("peak_amp",   "peak_amp_nmad"):     reasons.append("peak_amp_outlier")

    if rec["peak_amp"] < cuts["peak_amp_abs_min"]:       reasons.append("peak_amp_low")
    if rec["peak_amp"] > cuts["peak_amp_abs_max"]:       reasons.append("peak_amp_high")

    peak_tick_median = stats["peak_tick"]["median"]
    if abs(rec["peak_tick"] - peak_tick_median) > cuts["peak_tick_abs"]:
        reasons.append("peak_time_shift")

    if _nmad_shifted("charge", "charge_nmad"):           reasons.append("charge_outlier")

    if cuts.get("adc_abs_max") is not None:
        if abs(rec["raw_adc_min"]) > cuts["adc_abs_max"] or abs(rec["raw_adc_max"]) > cuts["adc_abs_max"]:
            reasons.append("raw_adc_saturation")

    if cuts.get("adc_min_threshold") is not None:
        if rec["raw_adc_min"] < cuts["adc_min_threshold"]:
            reasons.append("adc_negative_saturation")

    return reasons


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def build_wfset_quality(wfset, cuts=None, windows=None, analysis_label=DEFAULT_ANALYSIS_LABEL, verbose=True):
    """Compute quality metrics, evaluate cuts, return filtered WaveformSet.

    Parameters
    ----------
    wfset : WaveformSet
        Input waveform set (must have 'analysis_label' analysis run).
    cuts : dict, optional
        Override DEFAULT_QUALITY_CUTS. Merge with ``{**DEFAULT_QUALITY_CUTS, ...}``.
    windows : dict[str, slice], optional
        Override DEFAULT_QUALITY_WINDOWS.
    analysis_label : str
        Key of the BasicWfAna result that contains 'baseline'.
    verbose : bool
        Print progress and summary.

    Returns
    -------
    wfset_quality : WaveformSet
    stats_by_channel : dict
    pass_by_wf_id : dict[int, bool]
    reasons_by_wf_id : dict[int, list[str]]
    reason_counts : Counter
    channel_summary : dict
    """
    if cuts is None:
        cuts = DEFAULT_QUALITY_CUTS
    if windows is None:
        windows = DEFAULT_QUALITY_WINDOWS

    records            = []
    records_by_channel = defaultdict(list)

    if verbose:
        print("Computing quality metrics...")
    for wf_index, wf in enumerate(wfset.waveforms):
        if verbose and wf_index % 100_000 == 0:
            print(f"  {wf_index}/{len(wfset.waveforms)}", flush=True)
        rec = compute_quality_metrics(wf, wf_index, windows=windows, analysis_label=analysis_label)
        records.append(rec)
        records_by_channel[rec["channel_key"]].append(rec)

    stats_by_channel = compute_stats_by_channel(records_by_channel)

    pass_by_wf_id    = {}
    reasons_by_wf_id = {}
    reason_counts    = Counter()
    channel_summary  = defaultdict(lambda: {"total": 0, "pass": 0, "reject": 0})

    for rec in records:
        reasons = evaluate_quality_record(rec, stats_by_channel, cuts)
        passed  = len(reasons) == 0
        pass_by_wf_id[rec["wf_id"]]    = passed
        reasons_by_wf_id[rec["wf_id"]] = reasons
        ch_sum = channel_summary[rec["channel_key"]]
        ch_sum["total"] += 1
        if passed:
            ch_sum["pass"] += 1
        else:
            ch_sum["reject"] += 1
            reason_counts.update(reasons)

    def _select_quality(wf):
        return pass_by_wf_id.get(id(wf), False)

    if verbose:
        print("Building wfset_quality...")
    wfset_quality = WaveformSet.from_filtered_WaveformSet(wfset, _select_quality, show_progress=verbose)

    if verbose:
        n_total  = len(records)
        n_pass   = len(wfset_quality.waveforms)
        n_reject = n_total - n_pass
        print(f"Total: {n_total}  |  Pass: {n_pass} ({100*n_pass/n_total:.1f}%)  |  Reject: {n_reject}")
        print("Rejection reasons:")
        for reason, count in reason_counts.most_common():
            print(f"  {reason:25s}: {count}")

    return wfset_quality, stats_by_channel, pass_by_wf_id, reasons_by_wf_id, reason_counts, dict(channel_summary)
