# mpid2
Utilities for the D2 CVMC group at the Max Planck Institute for Informatics

## How-To

### Install

`python setup.py install` or `python setup.py develop`

### General note on dispatching on remote hosts

The underlying python scripts make use of a series of ssh calls. To circumvent repeated entering of the ssh password start an ssh agent:

`exec ssh-agent bash`
`ssh-add`

### Use pythond

- bash scripts/gpujob_dispatching/install_pythond.sh
- If `~/bin` is in `$PATH`, then pythond can be called anywhere
- To run a command on a free gpu/machine:
  `pythond num_gpus gpu_mem cpu_mem command.py --args`
  e.g. `pythond 1 8 0 train.py`, where memory is to be stated in GB.
- Set the number of required cpu memory to 0, if unknown
- To run a command on a free cpu machine set the number of required gpus to 0,
  e.g. `pythond 0 0 256`, to execute on cpu machines with at least 256GB of available RAM.

### Use parallel job dispatching

Create a python script that sends a list of terminal commands to the next free hosts:

```
from mpid2 import dispatcher

commands = [
  'cd myworkingdirectory; bash preprocessing.sh && python train.py',
  'cd myworkingdirectory; bash preprocessing.sh && python train_2.py',
  ]
  
 hosts = ['host1', 'host2', 'host3', 'host4']
 
 dispatcher.dispatch(hosts, commands, required_gpus=2, required_gpu_mem=8, required_cpu_mem=0, log_target='file')
```
