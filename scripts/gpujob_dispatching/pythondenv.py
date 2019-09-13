import os
import sys
from mpid2 import dispatcher

try:
	num_gpu = int(sys.argv[1])
	min_mem = int(sys.argv[2])
except:
	raise RuntimeError('First two arguments must be of type integer stating: required number of gpus and required amount of memory (in Megabytes)')

if min_mem > 11:
	#ignore = set([7, 10, 11, 15, 20, 21])
	ignore = set()
	HOSTS = ['d2volta%02d'%i for i in range(1, 23) if i not in ignore] + ['d2pascal%02d'%i for i in range(1, 5)]
elif min_mem < 6:
	HOSTS = ['wks-12-%d'%i for i in [31, 32, 33, 44, 45, 56, 47]]
else:
	HOSTS = ['menorca', 'sumatra', 'montreal', 'madagaskar', 'fiji', 'takatuka', 'greenland', 'kohtao', 'bermuda', 'jamaica', 'martinique', 'mauritius', 'lanzarote', 'jersey', 'costarica', 'borneo', 'reichenau', 'helgoland', 'helium']

cwd = os.getcwd()
environment = 'Python35'

command = 'cd %s; source activate %s; python %s'%(cwd, environment, ' '.join(sys.argv[3:]))

dispatcher.dispatch(HOSTS, [command], required_gpus=num_gpu, required_gpu_mem=min_mem, log_target='stdout')
