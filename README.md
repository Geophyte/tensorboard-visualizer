
# TensorBoard Visualizer

A Streamlit-based interactive app to visualize multiple TensorBoard runs grouped by user-defined tags extracted via regex. Supports aggregation (mean, median), individual run plotting, and interactive Plotly charts.

## Features

- Scan directories recursively for TensorBoard event files.
- Extract tags from file paths using regex with named groups.
- Group runs by tags and aggregate metrics.
- Plot metrics with mean/median and standard deviation/percentile bands.
- Optionally overlay individual runs.
- Interactive, fully customizable Plotly charts in Streamlit.

## Project Structure

tensorboard_visualizer/
│
└── src/
    ├── app.py                 # Main Streamlit app
    └── utils/
        │── __init__.py
        │── file_utils.py      # File scanning and path parsing
        │── event_utils.py     # TensorBoard event parsing and aggregation

````

## Installation

1. Clone the repository:

```bash
git clone <repo_url>
cd tensorboard_visualizer
````

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirenments
```

## Usage

Run the Streamlit app:

```bash
streamlit run src/app.py
```

Open the displayed **Local URL** (usually `http://localhost:8501`) in your browser.

### Sidebar Options

* **Data**

  * `Base directories`: directories to scan for TensorBoard events.
  * `Include regex` / `Exclude regex`: filter runs by path.
* **Tag extraction**

  * Regex patterns to extract tags from file paths (use named groups, e.g. `lr=(?P<lr>\d+\.\d+)`).
* **Plot**

  * Aggregation method: `mean` or `median`.
  * Show individual runs: overlay each run.
  * Order groups by: `none`, `best_final`, `best_max`.
  * Legend and margins customization.

## Example Usage

Suppose your TensorBoard logs are organized like this:

```

runs/
├── experiment1_lr=0.001_bs=32/
│   └── events.out.tfevents.123
├── experiment1_lr=0.001_bs=64/
│   └── events.out.tfevents.124
├── experiment2_lr=0.01_bs=32/
│   └── events.out.tfevents.125
└── experiment2_lr=0.01_bs=64/
    └── events.out.tfevents.126

```

You want to group runs by learning rate (`lr`) and batch size (`bs`) and visualize the `loss` metric.

### Regex Patterns

Enter the following in the **Tag extraction** textarea:

```
lr=(?P<lr>[\d.]+)_bs=(?P<bs>\d+)
```

This will extract:

| Path                                | lr     | bs  |
|------------------------------------|--------|-----|
| runs/experiment1_lr=0.001_bs=32/   | 0.001  | 32  |
| runs/experiment2_lr=0.01_bs=64/    | 0.01   | 64  |

### Grouping

- In **Group runs by these keys**, select `lr` and `bs`.  
- Each combination of `lr` and `bs` becomes a separate group in the plot.

### Metric

- Choose `loss` in the **Metric to plot** dropdown.  

### Result

- The app will plot the mean (or median) loss across all runs in each group.  
- Standard deviation bands will be shaded.  
- You can also overlay individual runs if `Show individual runs` is checked.

## License

MIT License
