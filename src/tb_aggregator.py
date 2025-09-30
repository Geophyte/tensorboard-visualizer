import streamlit as st
import os
import re
import pandas as pd
import numpy as np
from typing import List, Dict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(layout="wide", page_title="TensorBoard Grouped Visualizer")

# ----------------------- Helpers -----------------------

def find_event_files(base_dirs: List[str]) -> List[str]:
    matches = []
    for d in base_dirs:
        d = os.path.expanduser(d)
        for root, _, files in os.walk(d):
            for fname in files:
                if fname.startswith('events'):
                    matches.append(os.path.join(root, fname))
    return sorted(list(set(matches)))

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

# ----------------------- UI -----------------------

st.title('TensorBoard Grouped Visualizer (Regex Tags)')

with st.sidebar:
    st.header('Data')
    base_paths = st.text_area('Base directories (one per line)', value='.')
    base_dirs = [line.strip() for line in base_paths.splitlines() if line.strip()]
    include_pat = st.text_input('Include regex (applied to full path)', '')
    exclude_pat = st.text_input('Exclude regex (applied to full path)', '')
    
    st.header('Tag extraction')
    regex_input = st.text_area(
        'Regex patterns to extract tags (one per line, use named groups)\nExample: lr=(?P<lr>[\\d\\.]+)\nbs=(?P<bs>\\d+)',
        value=''
    )
    regex_patterns = [line.strip() for line in regex_input.splitlines() if line.strip()]

    st.header('Plot')
    agg_method = st.selectbox('Aggregation method', ['mean','median'])
    show_individual = st.checkbox('Show individual runs', value=False)
    order_by = st.selectbox('Order groups by', ['none','best_final','best_max'])
    legend_x = st.sidebar.slider('Legend X position', 0.0, 1.0, 0.01)
    legend_y = st.sidebar.slider('Legend Y position', 0.0, 1.0, 0.99)
    legend_bg_alpha = st.sidebar.slider('Legend background transparency', 0.0, 1.0, 0.5)
    margin_left = st.sidebar.number_input('Left margin', min_value=0, max_value=200, value=60)
    margin_right = st.sidebar.number_input('Right margin', min_value=0, max_value=200, value=60)
    margin_top = st.sidebar.number_input('Top margin', min_value=0, max_value=200, value=60)
    margin_bottom = st.sidebar.number_input('Bottom margin', min_value=0, max_value=200, value=60)

if st.button('Scan for runs'):
    event_files = find_event_files(base_dirs)
    st.session_state['event_files'] = event_files

files = st.session_state.get('event_files', [])
if not files:
    st.stop()

rows = []
for p in files:
    if include_pat and not re.search(include_pat, p):
        continue
    if exclude_pat and re.search(exclude_pat, p):
        continue
    md = parse_tags_from_path(p, regex_patterns)
    rows.append(md)

meta_df = pd.DataFrame(rows)
if meta_df.empty:
    st.warning('No runs matched filter')
    st.stop()

possible_keys = [c for c in meta_df.columns if not c.startswith('_')]
group_by = st.multiselect('Group runs by these keys', possible_keys, default=possible_keys[:1])

# Preview tags
preview_tags = set()
for p in meta_df['_path'].head(5):
    tags = parse_event_scalars(p)
    preview_tags.update(tags.keys())
metric = st.selectbox('Metric to plot', sorted(preview_tags))

meta_df['group'] = meta_df[group_by].astype(str).agg('|'.join, axis=1)
groups = meta_df.groupby('group')

results = []
fig = go.Figure()
for gname, gdf in groups:
    series_list = []
    for p in gdf['_path']:
        tags = parse_event_scalars(p)
        if metric in tags:
            series_list.append(tags[metric])
    x, Y = align_and_interpolate(series_list)
    c,l,u = aggregate_matrix(Y, method=agg_method)
    if c.size==0:
        continue
    label = f'{gname} (n={len(series_list)})'
    fig.add_trace(go.Scatter(x=x, y=c, mode='lines', name=label))
    fig.add_trace(go.Scatter(x=np.concatenate([x,x[::-1]]), y=np.concatenate([u,l[::-1]]), fill='toself', name=label+' band', opacity=0.2, showlegend=False))
    if show_individual:
        for s in series_list:
            fig.add_trace(go.Scatter(x=s['step'], y=s['value'], mode='lines', line={'dash':'dot'}, opacity=0.3, showlegend=False))
    final_val = c[-1]
    best_val = np.nanmax(c)
    results.append({'group':gname,'final':final_val,'best':best_val})

# ordering
if order_by != 'none' and results:
    dfres = pd.DataFrame(results)
    if order_by == 'best_final':
        dfres = dfres.sort_values('final', ascending=False)
    else:  # best_max
        dfres = dfres.sort_values('best', ascending=False)
else:
    dfres = pd.DataFrame(results)  # keep original order

ordered_groups = dfres['group'].tolist()

fig = go.Figure()
for gname in ordered_groups:
    gdf = groups.get_group(gname)
    series_list = []
    for p in gdf['_path']:
        tags = parse_event_scalars(p)
        if metric in tags:
            series_list.append(tags[metric])
    x, Y = align_and_interpolate(series_list)
    c, l, u = aggregate_matrix(Y, method=agg_method)
    if c.size == 0:
        continue
    
    final_val = c[-1]
    best_val = np.nanmax(c)
    value_display = final_val if order_by=='best_final' else best_val
    label = f"{gname} (n={len(series_list)}, val={value_display:.3f})"

    fig.add_trace(go.Scatter(
        x=x, y=c, mode='lines', name=label
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([x,x[::-1]]),
        y=np.concatenate([u,l[::-1]]),
        fill='toself',
        name=label+' band',
        opacity=0.2,
        showlegend=False
    ))

    if show_individual:
        for s in series_list:
            fig.add_trace(go.Scatter(
                x=s['step'],
                y=s['value'],
                mode='lines',
                line={'dash':'dot'},
                opacity=0.3,
                showlegend=False
            ))

fig.update_layout(
    height=700,
    hovermode='x unified',
    xaxis_title='step',
    yaxis_title=metric,
    margin=dict(l=margin_left, r=margin_right, t=margin_top, b=margin_bottom),
    legend=dict(
        yanchor="top",
        y=legend_y,
        xanchor="left",
        x=legend_x,
        bordercolor="Black",
        borderwidth=1,
        bgcolor=f'rgba(255,255,255,{legend_bg_alpha})'
    )
)
st.plotly_chart(fig, use_container_width=True)
st.dataframe(meta_df)
