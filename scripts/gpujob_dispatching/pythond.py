import os
import sys
from mpid2 import dispatcher

try:
	num_gpu = int(sys.argv[1])
	min_gpu_mem = int(sys.argv[2])
	min_cpu_mem = int(sys.argv[3])
except:
	raise RuntimeError('First three arguments must be of type integer stating: required number of gpus, required amount of GPU memory (in GB) and required amount of CPU memory (in GB)')

if num_gpu > 2 or min_gpu_mem > 11:
	ignore = set()
	HOSTS = ['d2volta%02d'%i for i in range(1, 22) if i not in ignore]
	HOSTS += ['d2pascal01', 'd2pascal02', 'd2pascal03', 'd2pascal04']
elif num_gpu > 0 and min_gpu_mem < 6:
	HOSTS = ['wks-12-%d'%i for i in [31, 32, 33, 44, 45, 56, 47]]
elif num_gpu > 0:
	HOSTS = ['menorca', 'sumatra', 'montreal', 'madagaskar', 'fiji', 'takatuka', 'greenland', 'kohtao', 'bermuda', 'jamaica', 'martinique', 'mauritius', 'lanzarote', 'jersey', 'costarica', 'borneo', 'reichenau', 'helgoland', 'helium']
else:
	# no gpus required
	ignore = set()
	HOSTS = ['d2blade%02d'%i for i in range(49) if i not in ignore]
	HOSTS = ['cuba', 'atlantis', 'kreta', 'capri'] + HOSTS

cwd = os.getcwd()

command = 'cd %s; python %s'%(cwd, ' '.join(sys.argv[4:]))

dispatcher.dispatch(HOSTS, [command], required_gpus=num_gpu, required_gpu_mem=min_gpu_mem, required_cpu_mem=min_cpu_mem, log_target='stdout')
