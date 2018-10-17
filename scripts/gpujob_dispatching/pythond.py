import os
import sys
from mpid2 import dispatcher

try:
	num_gpu = int(sys.argv[1])
	min_mem = int(sys.argv[2])
except:
	raise RuntimeError('First two arguments must be of type integer stating: required number of gpus and required amount of memory (in Megabytes)')

if num_gpu > 2:
	HOSTS = ['d2volta%02d'%i for i in range(1, 22)]
elif min_mem < 6000:
	HOSTS = ['wks-12-%d'%i for i in [31, 32, 33, 44, 45, 56, 47]]
else:
	HOSTS = ['menorca', 'samoa', 'sumatra', 'iceland', 'montreal', 'madagaskar', 'fiji', 'takatuka', 'greenland', 'kohtao', 'bermuda', 'jamaica', 'martinique', 'mauritius', 'lanzarote', 'jersey', 'costarica', 'borneo', 'reichenau', 'helgoland', 'helium', 'neon', 'argon']

cwd = os.getcwd()

command = 'cd %s; python %s'%(cwd, ' '.join(sys.argv[3:]))

dispatcher.dispatch(HOSTS, [command], required_gpus=num_gpu, required_mem=min_mem, log_target='stdout')
