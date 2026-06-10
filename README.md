# the gap

Code to reproduce the paper "Beyond Single-Factor Views of the Adam-SGD Gap".

This project is modular, meaning that each folder is self-contained.

### Setup

```bash
conda env create -f env.yml
conda activate gap
```

### Preparing the datasets

Each domain folder contains `prepare_*.py` scripts that download and preprocess data into `./data`. For example, for FineWeb-Edu (1B GPT-2 tokens, written to `data/fineweb/{train,val}.bin`):

```bash
python -m language.prepare_fineweb
```

### Running the experiments

Each folder has a `train.py` that runs the training given the arguments.

`run_fineweb.sh` is an example SLURM script that sweeps optimizer × learning rate × momentum on FineWeb. Metrics (train/val loss histories) are logged to Weights & Biases.

### Plotting

`_results/` contains the all the run metrics exported from wandb as CSVs; the provided notebooks read from there:

- `_plot_gap.ipynb`. Plots the Adam-SGD gap with lr and mom sweep.
- `_plot_figure1.ipynb`. Plots the main Figure 1: gap across all the setups
- `_plot_theory.ipynb`. Plots the theory model (self-contained)

### Credits

If you find this work useful, cite:

```bibtex
@article{zhang2026beyond,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {2026}
}
```

