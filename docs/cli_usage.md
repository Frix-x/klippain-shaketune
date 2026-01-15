# Shake&Tune CLI Usage

Shake&Tune includes a command-line interface (CLI) that allows you to generate graphs from existing measurement data without needing a running Klipper instance. This is particularly useful for processing data on a different machine than your printer or re-run a newer/older version of Shake&Tune on an already recorded data file. For integrated analysis in your printer using Klipper "macros", use the full Shake&Tune Klipper plugin instead of the CLI mode.

## Installation Requirements

The CLI mode uses the same dependencies as the main Shake&Tune plugin. Ensure you have:

- Python 3.9 or newer
- Required Python packages: `numpy`, `matplotlib`, `zstandard`
- The Klipper repository cloned locally in your home folder (no need to install it, just clone it)

You can install these dependencies using:
```bash
cd ~/
git clone https://github.com/Klipper3d/klipper.git
git clone https://github.com/Frix-x/klippain-shaketune.git
cd klippain-shaketune
pip install uv
uv pip install -r requirements.txt
```

## Basic Usage

The CLI follows this general pattern:
```bash
uv run python -m shaketune.cli <graph_type> [options] <input_files>
```

### Quick Reference

```bash
# Show available commands or detailed help for a specific command
python -m shaketune.cli --help
python -m shaketune.cli <graph_type> --help

# Example: Get help for input shaper graph generation command
python -m shaketune.cli input_shaper --help
```

### Available Graph Types

| Command | Description |
|---------|-------------|
| `static_freq` | Static frequency analysis |
| `axes_map` | Accelerometer axes mapping detection |
| `belts` | Belt tension comparison (CoreXY/CoreXZ) |
| `input_shaper` | Input shaper calibration |
| `vibrations` | Machine vibrations profile |

### Input File Formats

Shake&Tune CLI supports two input formats:

1. **`.stdata` files**: Shake&Tune's compressed binary format (recommended)
2. **`.csv` files**: Legacy Klipper raw accelerometer data files for compatibility with older versions of Shake&Tune

## Common Options

All commands support these common options:

- `-o, --output`: Output filename (required)
- `--max_freq`: Maximum frequency for analysis (Hz)
- `--dpi`: Graph resolution (default: 300)

## Command Examples

### 1. Axes Map Detection

```bash
# Using .stdata file
python -m shaketune.cli axes_map \
    -o ./results/axes_map_analysis.png \
    --accel 3000 \
    ./data/axesmap_20240817_212948.stdata

# Using CSV files (requires X, Y, Z measurements)
python -m shaketune.cli axes_map \
    -o ./results/axes_map_analysis.png \
    --accel 3000 \
    ./data/axesmap_*.csv

# With existing axes_map to verify/correct (use = syntax when value starts with -)
python -m shaketune.cli axes_map \
    -o ./results/axes_map_analysis.png \
    --accel 3000 \
    --axes_map="-y,x,z" \
    ./data/axesmap_20240817_212948.stdata
```

**Required parameters:**
- `--accel`: Acceleration used during measurement (mm/s²)

**Optional parameters:**
- `--axes_map`: Existing axes_map configuration to invert for analysis. Use this to verify or correct an already configured axes_map. **Important:** When the value starts with `-`, use the `=` syntax (e.g., `--axes_map="-y,x,z"`).

### 2. Static Frequency Analysis

```bash
python -m shaketune.cli static_freq \
    -o ./results/static_frequency.png \
    --frequency 45.0 \
    --duration 30.0 \
    --accel_per_hz 75.0 \
    ./data/staticfreq_20240817_*.stdata
```

**Optional parameters:** (these are only used for the legend and the title of the graph)
- `--frequency`: Maintained frequency during measurement (Hz)
- `--duration`: Duration of the measurement (seconds)
- `--accel_per_hz`: Acceleration per Hz used (mm/s²/Hz)

### 3. Belt Comparison (CoreXY/CoreXZ)

```bash
python -m shaketune.cli belts \
    -o ./results/belt_comparison.png \
    --kinematics corexy \
    --mode SWEEPING \
    --accel_per_hz 75.0 \
    --sweeping_accel 3000 \
    --sweeping_period 2.0 \
    --max_scale 50000 \
    -k ~/klipper \
    ./data/belts_20240818_*.stdata
```

**Required parameters:**
- `--kinematics`: Machine kinematics (`corexy`, `corexz`, `cartesian`, etc.)
- `-k, --klipper_dir`: Path to Klipper directory (for shaper calculations)

**Optional parameters:** (these are only used for the legend and the title of the graph)
- `--mode`: Test mode used (`SWEEPING`, `FIXED`, etc.)
- `--accel_per_hz`: Acceleration per Hz (mm/s²/Hz)
- `--sweeping_accel`: Acceleration for sweeping tests (mm/s²)
- `--sweeping_period`: Sweeping period (seconds)
- `--max_scale`: Maximum energy scale for graphs

### 4. Input Shaper Calibration

```bash
python -m shaketune.cli input_shaper \
    -o ./results/input_shaper_x.png \
    --scv 5.0 \
    --max_smoothing 0.15 \
    --mode SWEEPING \
    --accel_per_hz 75.0 \
    --sweeping_accel 3000 \
    --sweeping_period 2.0 \
    -k ~/klipper \
    ./data/inputshaper_X_*.stdata
```

**Required parameters:**
- `-k, --klipper_dir`: Path to Klipper directory
- `--scv`: Square corner velocity (mm/s)

**Optional parameters:** (these are only used for the legend and the title of the graph)
- `--max_smoothing`: Maximum allowed smoothing
- `--mode`: Test mode used
- `--accel_per_hz`: Acceleration per Hz (mm/s²/Hz)
- `--sweeping_accel`: Acceleration for sweeping tests (mm/s²)
- `--sweeping_period`: Sweeping period (seconds)
- `--max_scale`: Maximum energy scale for graphs

### 5. Vibrations Profile

```bash
python -m shaketune.cli vibrations \
    -o ./results/vibrations_profile.png \
    --kinematics corexy \
    --accel 7000 \
    -k ~/klipper \
    ./data/vibrations_*.stdata
```

**Required parameters:**
- `--kinematics`: Machine kinematics
- `--accel`: Acceleration used during measurements (mm/s²)
- `-k, --klipper_dir`: Path to Klipper directory
