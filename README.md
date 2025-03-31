# Yeast_Galactose_Switch_RDMEODE
code for yeast galactose switch paper 

## Contents
+ `Lattice-Microbes_YeastRDMEODE`: software for simulation with installation guide. 
+ `cmeode`: code for hybrid CME-ODE simulation.
+ `rdmeode`： code for hybrid RDME-ODE simulation

## How to run 

### 1. install the Lattice-Mircrobes via conda

Go to the folder: 

```bash
cd Lattice-Microbes_YeastRDMEODE
```

#### Prerequisites
Install Anaconda 3 if not already installed on your computer, download and follow the instructions at
https://www.anaconda.com/

If you want to do spatial RDME simulations which require GPUs, install CUDA, download and instructions can be found at
https://developer.nvidia.com/cuda-downloads

Add the following line at the end of /.bashrc so LM can find CUDA.  **If path to CUDA is different**:

```bash
export path to where CUDA is installed:
export PATH="/usr/local/cuda/bin/:$PATH"
```

#### LM installation via conda
The conda environment can be created from the yml file included here.  Go into the file and change "lmpretest" to whatever you want your environment name to be, then run:

replace `envname` with your choice of name for the environment.

```bash
    conda env create -n lm2.5_dev conda_envs/lm_precomp.yml
    
    conda activate lm2.5_dev
```

Once you have created and activated your environment, go to a directory where you want to build Lattice Microbes and do the build running the following commands:
(You do not need to run simulations where the build is located, lm will be installed in your environment as an executable)


create a build directory and build with cmake:

```bash
mkdir build
cd build
cmake ../src/ -D MPD_GLOBAL_T_MATRIX=True -D MPD_GLOBAL_R_MATRIX=True
# The -D flags can be removed if you build with smaller transition matrix and reaction matrix size limits in the cmake options.
make && make install
# Run make install as root or sudo if possible
```

> ⚠️ **Warning**: Manually set `CUDA_ARCHITECTURES` if needed:
>
> If you want to install as a docker container or apptainer, look at the README.md fiile in the `Lattice-Microbes_YeastRDMEODE` folder.


### 2. Run hybrid CME-ODE simulation

```bash 
cd cmeode
bash ode_cme.sh
```

### 3. Run hybrid RDME-ODE simulation

```bash
cd rdmeode/main_code
bash ode_rdme.sh
```