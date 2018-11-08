# mpid2
Utilities for the D2 CVMC group at the Max Planck Institute for Informatics

## How-To

### Install

`python setup.py install` or `python setup.py develop`

### Use pythond

- bash scripts/gpujob_dispatching/install_pythond.sh
- If `~/bin` is in `$PATH`, then pythond can be called anywhere
- To run a command on a free gpu/machine:
  `pythond num_gpus min_mem command.py --args`
  e.g. `pythond 1 8000 train.py`

### Use parallel job dispatching

Create a python script that sends a list of terminal commands to the next free hosts:

```
from mpid2 import dispatcher

commands = [
  'cd myworkingdirectory; bash preprocessing.sh && python train.py',
  'cd myworkingdirectory; bash preprocessing.sh && python train_2.py',
  ]
  
 hosts = ['host1', 'host2', 'host3', 'host4']
 
 dispatcher.dispatch(hosts, commands, required_gpus=2, required_mem=8000, log_target='file')
```
