## Test Code

You need first change the location of the SUMO installment at ```traffic_envs.config.py``` line 8, 
```
os.environ["SUMO_HOME"] = "$your_sumo_path"
```

Run the following code to see the traffic simulation environment:

```
python env_test.py
```

## Installment of Dependents

### Dependents of this project

Create new conda env (see more for [conda cmds](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))::

```
conda create -n $env_name python=3.8
conda activate $env_name
```

Install dependents of the project:

```
pip install e .
```

I also put a max-pressure controller in this environment as an example controller. This controller uses the PyTorch package, you should download the [PyTorch](https://pytorch.org/get-started/locally/) to use the max-weight controller. 

### Installment of SUMO

### Windows system

Directly download the sumo from the official website [SUMO](https://www.eclipse.org/sumo/). 

> The official python api of SUMO is [TraCI](https://sumo.dlr.de/docs/TraCI.html). It uses a TCP-based server for the user to interact with the simulation environment. However, this API is very slow there are too many vehicles in the network. [Libsumo](https://sumo.dlr.de/docs/Libsumo.html) is almost the same with TraCI for the usage but is far more efficient than TraCI. However, the installment of Libsumo in Windows is not tested yet.


### Installment of SUMO (with Libsumo) in Linux

Replacing the libsumo with traci can speed up the communication between the simulation and the python project about 10 times (for this project). 

Currently I only tried the installment in Linux system (Ubuntu). The installment document is available at [Libsumo](https://sumo.dlr.de/docs/Libsumo.html) (Linux: [Libsumo for Linux](https://sumo.dlr.de/docs/Installing/Linux_Build.html)). Here is what I installed the libsumo, which is slightly different from the document provided in the previous link:


Create and activate a new conda env (see more for [conda cmds](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)):

```
conda create -n sumo python=3.8
conda activate sumo
```

Install all of the required tools and libraries

```
sudo apt-get install cmake python g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev swig
```
Get the source code (there is no requirement for the path of this downloaded code, this is only used for this installing):
```
git clone --recursive https://github.com/eclipse/sumo
```
Build the SUMO binaries:
```
export SUMO_HOME="$PWD/sumo"
mkdir sumo/build/cmake-build && cd sumo/build/cmake-build
```
Then the following cmd is different from the docs provided in the official link ([Libsumo for Linux](https://sumo.dlr.de/docs/Installing/Linux_Build.html)). The reason is that here we choose to install the sumo to the created conda env:
```
cmake ../.. -DCMAKE_INSTALL_PREFIX=$env_path
```
where ```$env_path``` is the path to the created conda env (usually ```~/Anaconda/envs/$env_name```). 

Finally, run the following cmds:
```
make -j$(nproc)
make install
```
To verify the installment:

```
python
import traci
import libsumo
```

#### (Optional after installment) Define the env variable

To predefine the env variable (you need to define the env variable for each time), you can choose either way:

1. Change the configuration file of the terminal (can only used for running code directly at terminal). Add the definitation of the env variable to the file ```~/.bashrc```. You can use any editor in terminal including ```nano``` or ```gedit```. For example:

```
gedit ~/.bashrc
```
  Then add the following sentence (any row) to the file ```~/.bashrc```:
```
export SUMO_HOME="$env_path"
```
2. (Recommend) Add the following code to the python project before importing sumo libs:

```
os.environ["SUMO_HOME"] = "$env_path"
```


