# RDME-ODE Simulation

This directory contains the main simulation code for the combined RDME/ODE galactose switch model.

## Quick Start

For local execution, refer to [ode_rdme.sh](ode_rdme.sh) or run `python galactose_rdmeode_combined.py -h` for help. HPC job submission commands are documented in [dtai_multi_run.sh](./HPC_slurm/dtai_multi_run.sh).

## Command-Line Parameters

| Argument                      | Type  | Default                  | Description                                                |
| ----------------------------- | ----- | ------------------------ | ---------------------------------------------------------- |
| `-id`, `--index`              | int   | _required_               | Index of the output `.lm` files                            |
| `-t`, `--simtime`             | float | 60                       | Simulation time (minutes)                                  |
| `-g`, `--galactose`           | float | 11.1                     | External galactose concentration (mM)                      |
| `-gpus`, `--gpus`             | int   | 1                        | Number of GPUs to use                                      |
| `-tag`, `--tag`               | str   | `''`                     | Tag for the output folder                                  |
| `-geo`, `--geometry`          | str   | `yeast-lattice.2.pkl.xz` | Geometry file name                                         |
| `-mt`, `--max_time`           | float | 1000                     | Maximum allowed wall-clock time (hours)                    |
| `-geloc`, `--gene_location`   | str   | `random`                 | Gene location: `random`, `center`, `edge`, or `chromosome` |
| `-ckpt`, `--checkpoint`       | str   | `''`                     | Checkpoint file for resuming simulations                   |
| `--enable-chromosome`         | flag  | False                    | Enable chromosome regions                                  |
| `--enable-er`                 | flag  | False                    | Enable ER regions                                          |
| `-ernum`, `--er_num`          | int   | 4                        | Number of ER tunnels                                       |
| `--enable-effective-ribosome` | flag  | False                    | Enable effective ribosome model                            |
| `--enable_rna_tracking`       | flag  | False                    | Enable RNA tracking                                        |

## Usage Examples

### Baseline Model

```bash
python galactose_rdmeode_combined.py -id 99 -t 60 -g 11.1 -gpus 2 \
    -geo "lattice_ribosomes_noER_345964_isolated.pkl.xz"
```

### With Chromosome

```bash
python galactose_rdmeode_combined.py -id 99 -t 60 -g 11.1 -gpus 2 \
    -geo "lattice_ribosomes_noER_345964_isolated.pkl.xz" \
    --enable-chromosome -geloc chromosome
```

### With Chromosome + ER

```bash
python galactose_rdmeode_combined.py -id 99 -t 60 -g 11.1 -gpus 2 \
    -geo "lattice_ribosomes_ER_345964_isolated.pkl.xz" \
    --enable-chromosome -geloc chromosome --enable-er
```

### With Chromosome + ER + Effective Ribosome

```bash
python galactose_rdmeode_combined.py -id 99 -t 60 -g 11.1 -gpus 2 \
    -geo "lattice_ribosomes_ER_345964_isolated.pkl.xz" \
    --enable-chromosome -geloc chromosome --enable-er --enable-effective-ribosome
```

### Resuming from Checkpoint

To extend a simulation (e.g., from 1 hour to 7 hours), specify the checkpoint file from the previous run:

```bash
python galactose_rdmeode_combined.py -id 99 -t 3 -g 11.1 -gpus 2 \
    -geo "lattice_ribosomes_ER_345964_isolated.pkl.xz" \
    --enable-chromosome --enable-er \
    -ckpt "../rdme_ode_results/20251204_ER1121_extension/0_1h/yeast1.17_combined_20251120_104_t60.0min_GAE11.1mM_ER_CHROMOchromo_ER_345k_ERribo_gpu4.lm"
```
