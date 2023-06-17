# Shared Memory Programming with MPI



# Usage
## build the program
```bash
mkdir out
make
```
## Run the program
run the program with default configuration
```bash
mpiexec -n 20 ./nsexe
```
run the program with custom configuration
```bash
mpiexec -n 20 ./nsexe user_config
```
### parameters
  suppose the decomposition along x is b and the domain dimension along x is a. then 
  ```b-1<a%b+floor(a/b)```
  must be satisfied for the decomposition to work. Because for load balancing each process is doing 1 extra work and there need to be enough leftover points in the last block for the rest of the block to take. In actual work this shouldn't be a problem as each process should have a large amount of work to work on.
### user_config file
the user_config file defines the configuration of the simulation, the format is as follows
```
Nx 201
Ny 101
Lx 0.1
Ly 0.05
rho 1000
nu 1e-6
P_max 0.5
t_end 50.0
dt_min 1.e-3
courant 0.01
dt_out 0.5
```
the first column represent the name of variable in code, the second row is the value of the variable, seperate by a single space. The file can have missing varaibles or extra variables.
### output file
the output file are contained in the `out` directory. the filename format is `id_0_P_1.dat` where `0` is the process id, `1` is the timestep, `P` is the field.
The output file also have a file called `config_file.dat` which contains useful information for post procesing. To successfully generate the output, the user must make sure to have the `out` directory. hpc log are in `hpcfiles`.
## Post processing
Post processing is done using python code. `postprocessing.ipynb` is the jupyter notebook version and `postprocessing.py` is the script version. The final result is generate in three directories. `gifs` contains the generated gifs of different feilds. `img` contains the individual images that make up the gifs(i.e gif frames). `result_data` is the data for the whole domain.

