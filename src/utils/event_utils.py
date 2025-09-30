import pandas as pd
from typing import Dict, List
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import streamlit as st
import re


@st.cache_data(show_spinner=False)
def parse_event_scalars(path: str) -> Dict[str, pd.DataFrame]:
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    try:
        ea.Reload()
    except Exception:
        return {}
    tags = ea.Tags().get('scalars', [])
    out = {}
    for tag in tags:
        try:
            events = ea.Scalars(tag)
        except Exception:
            continue
        steps = [e.step for e in events]
        vals = [e.value for e in events]
        df = pd.DataFrame({'step': steps, 'value': vals})
        out[tag] = df
    return out

def parse_tags_from_path(path: str, regex_patterns: List[str]) -> Dict[str, str]:
    """Extract named groups from path using user-defined regex patterns."""
    md = {}
    for pattern in regex_patterns:
        try:
            match = re.search(pattern, path)
            if match:
                md.update({k:v for k,v in match.groupdict().items() if v is not None})
        except re.error:
            continue  # skip invalid patterns
    md['_path'] = path
    return md


def align_and_interpolate(series_list: List[pd.DataFrame]):
    all_steps = np.unique(np.concatenate([s['step'].to_numpy() for s in series_list if len(s)>0]))
    if len(all_steps) == 0:
        return np.array([]), np.empty((0,0))
    if len(all_steps) > 500:
        all_steps = np.linspace(all_steps.min(), all_steps.max(), 500)
    Y = []
    for s in series_list:
        if len(s)==0:
            continue
        Y.append(np.interp(all_steps, s['step'], s['value'], left=np.nan, right=np.nan))
    return all_steps, np.vstack(Y)


def aggregate_matrix(Y: np.ndarray, method: str = 'mean'):
    if Y.size == 0:
        return np.array([]), np.array([]), np.array([])
    if method == 'mean':
        c = np.nanmean(Y, axis=0)
        l = c - np.nanstd(Y, axis=0)
        u = c + np.nanstd(Y, axis=0)
    elif method == 'median':
        c = np.nanmedian(Y, axis=0)
        l = np.nanpercentile(Y, 25, axis=0)
        u = np.nanpercentile(Y, 75, axis=0)
    else:
        c = np.nanmean(Y, axis=0)
        l = np.nanpercentile(Y, 25, axis=0)
        u = np.nanpercentile(Y, 75, axis=0)
    return c, l, u
