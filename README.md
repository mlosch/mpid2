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
