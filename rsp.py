"""
Respiratory signal processing and quality assessment functions based on NeuroKit2.
"""

import pandas as pd
import numpy as np
import neurokit2 as nk
from scipy import stats
from scipy.stats import pearsonr


def rsp_processing(rsp_signal, epoch_min=1, sampling_rate=60, method='khodadad2018',
                   quality_function=None, padding_sec=5, **quality_kwargs):
    '''
    Respiratory signal processing function with padding

    - Processes respiratory signal in epochs using NeuroKit2 `nk.rsp_process()` function
    - Default epoch length is 1 minute but can be adjusted
    - Each epoch is processed with padding before and after to avoid edge effects
    - Optionally computes quality metrics for each epoch using a provided quality function
    - Peaks and troughs are extracted from each epoch and adjusted to global indices
    - Returns signals DataFrame and info dictionary with concatenated results

    Parameters
    ----------
    rsp_signal : array-like
        Raw respiratory waveform signal
    epoch_min : float, optional
        Length of each epoch in minutes. Default is 1.
    sampling_rate : int, optional
        Sampling rate in Hz. Default is 60.
    method : str, optional
        Method for RSP processing. Default is 'khodadad2018'.
    quality_function : callable, optional
        Function to compute quality metrics. Should accept (signals, info, sampling_rate) and return a single boolean value for the epoch.
        For example: rsp_signal_quality
    padding_sec : float, optional
        Number of seconds to pad before and after each epoch. Default is 5.
    **quality_kwargs : dict, optional
        Additional keyword arguments to pass to quality_function

    Returns
    -------
    signals : pd.DataFrame
        DataFrame containing:
        - RSP_Epoch: epoch number
        - RSP_Epoch_Start: start time of epoch in seconds
        - RSP_Raw: raw signal
        - RSP_Clean: cleaned signal
        - RSP_Rate: respiratory rate
        - RSP_Phase: respiratory phase
        - RSP_PhaseCompletion: phase completion
        - RSP_Amplitude: amplitude
        - Acceptable (if quality_function provided AND quality_kwargs not included or specifies detailed=False): True indicates acceptable quality, False indicates unacceptable
    info : dict
        Dictionary containing:
        - RSP_Peaks: concatenated peak indices
        - RSP_Troughs: concatenated trough indices
        - sampling_rate: sampling rate
        - method: processing method
    all_quality : pd.DataFrame, optional
        DataFrame containing detailed quality metrics for each epoch if quality_function provided AND quality_kwargs specifies detailed=True
    '''
    # Divide signal into epochs
    signal_length = len(rsp_signal)
    epoch_length = int(epoch_min * 60 * sampling_rate)  # minutes converted to samples
    padding_samples = int(padding_sec * sampling_rate)  # padding in samples
    num_epochs = signal_length // epoch_length  # full length epochs

    # Initialize storage for concatenated results
    all_signals = []
    all_peaks = []
    all_troughs = []
    acceptability = np.array([])
    all_quality = pd.DataFrame()

    # Loop through epochs
    for i in range(num_epochs):
        # Define core epoch boundaries
        start_idx = i * epoch_length
        end_idx = start_idx + epoch_length

        # Define padded boundaries (with bounds checking)
        padded_start = max(0, start_idx - padding_samples)
        padded_end = min(signal_length, end_idx + padding_samples)

        # Extract padded signal
        padded_signal = rsp_signal[padded_start:padded_end]

        # Calculate offset for trimming processed signal back to core epoch
        offset_start = start_idx - padded_start
        offset_end = offset_start + epoch_length

        # Process the padded epoch
        epoch_signals_padded, epoch_info_padded = nk.rsp_process(
            padded_signal,
            sampling_rate=sampling_rate,
            method=method
        )

        # Extract only the core epoch from padded results
        epoch_signals = epoch_signals_padded.iloc[offset_start:offset_end].copy()
        epoch_signals.reset_index(drop=True, inplace=True)

        # Add epoch metadata
        epoch_signals['RSP_Epoch'] = i + 1
        epoch_signals['RSP_Epoch_Start'] = epoch_min * i * 60  # in seconds
        epoch_signals['RSP_Raw'] = rsp_signal[start_idx:end_idx]

        # Filter peaks and troughs to only those in the core epoch (relative to padded signal)
        peaks_in_epoch = epoch_info_padded['RSP_Peaks']
        peaks_in_epoch = peaks_in_epoch[(peaks_in_epoch >= offset_start) & (peaks_in_epoch < offset_end)]

        troughs_in_epoch = epoch_info_padded['RSP_Troughs']
        troughs_in_epoch = troughs_in_epoch[(troughs_in_epoch >= offset_start) & (troughs_in_epoch < offset_end)]

        # Adjust peak and trough indices to global indices
        epoch_peaks = peaks_in_epoch - offset_start + start_idx
        epoch_troughs = troughs_in_epoch - offset_start + start_idx

        # Create epoch_info for quality assessment (with local indices)
        epoch_info = {
            'RSP_Peaks': peaks_in_epoch - offset_start,
            'RSP_Troughs': troughs_in_epoch - offset_start
        }

        # Compute quality if function provided
        if quality_function is not None:
            try:
                epoch_quality = quality_function(epoch_signals, epoch_info, sampling_rate=sampling_rate, **quality_kwargs)
                if isinstance(epoch_quality, dict):
                    # detailed=True: quality_function returned a dict of features
                    if all_quality.empty:
                        all_quality = pd.DataFrame([epoch_quality])
                    else:
                        all_quality = pd.concat([all_quality, pd.DataFrame([epoch_quality])], ignore_index=True)
                elif isinstance(epoch_quality, str):
                    # detailed=False: quality_function returned 'acceptable' or 'unacceptable'
                    val = epoch_quality == 'acceptable'
                    epoch_accept = np.repeat(val, epoch_length)
                    acceptability = np.concatenate([acceptability, epoch_accept]) if acceptability.size else epoch_accept
                else:
                    # Boolean or other scalar return
                    epoch_accept = np.repeat(epoch_quality, epoch_length)
                    acceptability = np.concatenate([acceptability, epoch_accept]) if acceptability.size else epoch_accept
            except Exception as e:
                print(f"Quality assessment failed for epoch {i+1}: {e}")
                # Append default unacceptable quality so acceptability stays aligned with signals
                epoch_accept = np.repeat(False, epoch_length)
                acceptability = np.concatenate([acceptability, epoch_accept]) if acceptability.size else epoch_accept

        # Store results
        all_signals.append(epoch_signals)
        all_peaks.append(epoch_peaks)
        all_troughs.append(epoch_troughs)

    # Concatenate all epochs
    signals = pd.concat(all_signals, axis=0, ignore_index=True)

    # Label quality metric epochs
    if not all_quality.empty:
        all_quality = all_quality.reset_index(drop=False).rename(columns={'index': 'Epoch Start'}).reset_index(drop=False).rename(columns={'index': 'Epoch'})
        all_quality['Epoch'] = all_quality['Epoch'] + 1
        all_quality['Epoch Start'] = all_quality['Epoch Start'] * epoch_min * 60

    # Concatenate peaks and troughs
    rsp_peaks = np.concatenate(all_peaks) if all_peaks else np.array([])
    rsp_troughs = np.concatenate(all_troughs) if all_troughs else np.array([])

    # Prepare output info dictionary
    info = {
        'RSP_Peaks': rsp_peaks,
        'RSP_Troughs': rsp_troughs,
        'sampling_rate': sampling_rate,
        'method': method
    }

    # Return with quality information if computed
    if quality_function is None:
        return signals, info
    elif 'detailed' in quality_kwargs and quality_kwargs['detailed'] == True:
        return signals, info, all_quality
    else:
        signals['Acceptable'] = acceptability
        return signals, info


def rsp_quality_tree(rr_signals, rr_info, sampling_rate=60, detailed=False):
    """
    Classify respiratory signal quality from processed signal data (segment-level kurtosis model).

    This function calculates necessary SQI features from the processed respiratory signal
    and applies a decision tree classifier using only segment-level kurtosis (no breath-level).

    Parameters
    ----------
    rr_signals : pd.DataFrame
        Processed RSP signals DataFrame containing 'RSP_Clean' column
    rr_info : dict
        Info dictionary containing 'RSP_Peaks' and optionally 'RSP_Troughs'
    sampling_rate : int, optional
        Sampling rate in Hz. Default is 60.
    detailed : bool, optional
        If True, returns dict with all computed features and classification.
        If False, returns only 'acceptable' or 'unacceptable' string.
        Default is False.

    Returns
    -------
    str or dict
        If detailed=False: 'acceptable' or 'unacceptable'
        If detailed=True: Dictionary containing:
            - 'classification': 'acceptable' or 'unacceptable'
            - 'Breath Ratio': float
            - 'Template Correlation': float
            - 'Duration CV': float
            - 'Kurtosis_seg': float
            - 'n_breaths': int
            - 'respiratory_rate': float

    Notes
    -----
    Decision tree thresholds (segment-level kurtosis model):
        - Template Correlation <= 0.888738:
            - Template Correlation <= 0.847977 → unacceptable
            - else: Kurtosis_seg <= -0.705427 → acceptable, else unacceptable
        - Template Correlation > 0.888738:
            - Kurtosis_seg <= -0.321581 → acceptable
            - else: Duration CV <= 0.174857 → acceptable
            - else: Breath Ratio <= 0.973056 → acceptable, else unacceptable
    """
    # Extract signal and peaks
    cleaned_signal = rr_signals['RSP_Clean'].to_numpy()
    peaks = rr_info['RSP_Peaks']

    # Initialize features dict
    features = {
        'Breath Ratio': np.nan,
        'Template Correlation': np.nan,
        'Duration CV': np.nan,
        'Kurtosis_seg': np.nan,
        'n_breaths': 0,
        'respiratory_rate': np.nan
    }

    # =========================================================================
    # Calculate Kurtosis_seg (whole segment)
    # =========================================================================
    features['Kurtosis_seg'] = stats.kurtosis(cleaned_signal)

    # =========================================================================
    # Calculate Breath Ratio
    # =========================================================================
    ibi = np.diff(peaks)

    if len(ibi) > 0:
        breaths_expected = len(cleaned_signal) / np.median(ibi)
        breaths_actual = len(peaks)
        features['Breath Ratio'] = breaths_actual / breaths_expected
    else:
        features['Breath Ratio'] = 0.0

    # =========================================================================
    # Check for sufficient peaks
    # =========================================================================
    if len(peaks) < 3:
        features['classification'] = 'unacceptable'

        if detailed:
            return features
        return 'unacceptable'

    # =========================================================================
    # Segment into individual breaths (peak-to-peak) for template correlation
    # =========================================================================
    breaths_normalized = []

    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i + 1]
        if end <= start or end > len(cleaned_signal):
            continue

        breath = cleaned_signal[start:end]

        # Normalize duration (resample to fixed length of 100 points)
        breath_resampled = np.interp(
            np.linspace(0, 1, 100),
            np.linspace(0, 1, len(breath)),
            breath
        )

        # Normalize amplitude
        breath_range = breath_resampled.max() - breath_resampled.min()
        if breath_range > 1e-10:
            breath_normalized = (breath_resampled - breath_resampled.min()) / breath_range
            breaths_normalized.append(breath_normalized)

    features['n_breaths'] = len(breaths_normalized)

    if len(breaths_normalized) < 2:
        features['classification'] = 'unacceptable'

        if detailed:
            return features
        return 'unacceptable'

    # =========================================================================
    # Calculate Template Correlation
    # =========================================================================
    template = np.mean(breaths_normalized, axis=0)
    correlations = [np.corrcoef(b, template)[0, 1] for b in breaths_normalized]
    features['Template Correlation'] = np.mean(correlations)

    # =========================================================================
    # Calculate Duration CV
    # =========================================================================
    breath_durations = np.diff(peaks) / sampling_rate
    if len(breath_durations) > 1:
        features['Duration CV'] = np.std(breath_durations) / np.mean(breath_durations)
        features['respiratory_rate'] = 60 / np.mean(breath_durations)

    # =========================================================================
    # Apply Decision Tree Classification Logic (segment-level kurtosis model)
    # =========================================================================
    if features['Template Correlation'] <= 0.888738:
        if features['Template Correlation'] <= 0.847977:
            classification = 'unacceptable'
        else:
            if features['Kurtosis_seg'] <= -0.705427:
                classification = 'acceptable'
            else:
                classification = 'unacceptable'
    else:
        if features['Kurtosis_seg'] <= -0.321581:
            classification = 'acceptable'
        else:
            if features['Duration CV'] <= 0.174857:
                classification = 'acceptable'
            else:
                if features['Breath Ratio'] <= 0.973056:
                    classification = 'acceptable'
                else:
                    classification = 'unacceptable'

    features['classification'] = classification

    if detailed:
        return features
    return classification


def rsp_quality_charlton(epoch_signals, epoch_info, sampling_rate=60, detailed=False):
    """
    Compute respiratory SQI for a single epoch using Charlton et al. metrics,
    compatible with rsp_processing().

    Parameters
    ----------
    epoch_signals : pd.DataFrame
        NeuroKit2 signals DataFrame for this epoch. Must contain 'RSP_Clean'.
    epoch_info : dict
        Must contain 'RSP_Peaks' and 'RSP_Troughs' as arrays of sample indices
        (0-indexed within the epoch).
    sampling_rate : int
        Sampling rate in Hz.
    detailed : bool, optional
        If True, returns dict with all computed features and classification.
        If False, returns only 'acceptable' or 'unacceptable' string.
        Default is False.

    Returns
    -------
    str or dict
        If detailed=False: 'acceptable' or 'unacceptable'
        If detailed=True: Dictionary containing:
            - 'classification': 'acceptable' or 'unacceptable'
            - 'prop_norm_dur': float
            - 'prop_bad_breaths': float
            - 'R2': float (mean template correlation)
            - 'R2min': float (CV of cycle durations)
    """
    # Default features dict for detailed mode
    features = {
        'classification': 'unacceptable',
        'prop_norm_dur': 0.0,
        'prop_bad_breaths': 100.0,
        'R2': 0.0,
        'R2min': 0.0,
    }

    signal = epoch_signals['RSP_Clean'].to_numpy()
    time = np.arange(len(signal)) / sampling_rate  # seconds
    rel_peaks = np.asarray(epoch_info['RSP_Peaks'], dtype=int)

    # Need at least 2 peaks to compute breath-to-breath intervals
    if len(rel_peaks) < 2:
        if detailed:
            return features
        return 'unacceptable'

    # Cycle durations in seconds (peak-to-peak intervals)
    valid_cycle_durations = np.diff(rel_peaks) / sampling_rate

    # Check for degenerate durations before proceeding
    if len(valid_cycle_durations) == 0 or np.mean(valid_cycle_durations) == 0:
        if detailed:
            return features
        return 'unacceptable'

    # --- Template construction (same logic as MATLAB original) ---
    rr = int(np.floor(np.mean(np.diff(rel_peaks))))  # mean interval in samples
    half_rr = int(np.floor(rr / 2))

    j = np.where(rel_peaks > half_rr)[0]
    l = np.where(rel_peaks + half_rr < len(signal))[0]

    if len(j) == 0 or len(l) == 0:
        if detailed:
            return features
        return 'unacceptable'

    new_rel_peaks = rel_peaks[j[0]: l[-1] + 1]

    if len(new_rel_peaks) == 0:
        if detailed:
            return features
        return 'unacceptable'

    ts = []
    for peak in new_rel_peaks:
        t = signal[peak - half_rr: peak + half_rr + 1]
        norm_val = np.linalg.norm(t)
        tt = t / norm_val if norm_val > 0 else t
        ts.append(tt.flatten())

    ts = np.array(ts)
    avtempl = np.mean(ts, axis=0) if ts.shape[0] > 1 else np.full(ts.shape[1], np.nan)

    # --- Correlation of each breath with the template ---
    r2 = np.array([
        pearsonr(avtempl, ts[k])[0] if not np.any(np.isnan(avtempl)) else np.nan
        for k in range(ts.shape[0])
    ])
    R2 = float(np.nanmean(r2))

    # --- Coefficient of variation of cycle durations ---
    R2min = float(np.std(valid_cycle_durations) / np.mean(valid_cycle_durations))

    # --- Flag abnormal breath durations ---
    median_dur = np.median(valid_cycle_durations)
    bad_mask = (
        (valid_cycle_durations > 1.5 * median_dur) |
        (valid_cycle_durations < 0.5 * median_dur)
    )
    prop_bad_breaths = float(100 * np.sum(bad_mask) / len(bad_mask))

    # --- Proportion of window filled by normal-duration breaths ---
    norm_dur = float(np.sum(valid_cycle_durations[~bad_mask]))
    win_length = time[-1] - time[0]
    prop_norm_dur = float(100 * norm_dur / win_length) if win_length > 0 else 0.0

    # --- Quality decision ---
    qual = bool(
        prop_norm_dur > 60 and
        prop_bad_breaths < 15 and
        R2 >= 0.75 and
        R2min < 0.25
    )

    classification = 'acceptable' if qual else 'unacceptable'

    if detailed:
        return {
            'classification': classification,
            'prop_norm_dur': prop_norm_dur,
            'prop_bad_breaths': prop_bad_breaths,
            'R2': R2,
            'R2min': R2min,
        }
    return classification
