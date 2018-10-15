import subprocess
import time
import sys
import datetime
from functools import partial
from random import shuffle, seed
import os

MAX_PARALLEL_JOBS = 4

"""
How often does a job tries to rerun a job if it failed
"""
MAX_RETRIES = 3


def _print_info(info):
	print('[I]: '+info)

def _print_warning(err):
	print('\033[93m[E]: ' + err + '\033[0m')

def _print_error(err):
	print('\033[91m\033[1m[E]: ' + err + '\033[0m')	

def _print_ok(info):
	print('\033[92m[I]: ' + info + '\033[0m')


def time_stamped(fmt='%Y-%m-%d-%H-%M-%S.%f'):
    return datetime.datetime.now().strftime(fmt)[:-3]


def call(host, command, logfile=None):
	p = subprocess.Popen(['ssh', host, command], shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	output = []
	with p.stdout:
	    for line in iter(p.stdout.readline, b''):
	    	if logfile is not None:
	    		logfile.write(line)
	    		logfile.flush()

	    		output = line
	    	else:
	        	output.append(line.strip())
	p.wait()

	rc = p.returncode

	return rc, output

def query_gpu_utilization(host):
	cmd = 'nvidia-smi --query-gpu="memory.used,memory.total" --format=csv,noheader,nounits'
	retcode, output = call(host, cmd)

	if retcode != 0:
		raise RuntimeError('Could not query gpu info via nvidia-smi')

	num_gpus = len(output)
	mem_util = []
	for gpu in output:
		mem_used, mem_total = gpu.split(', ')
		mem_util.append((int(mem_used), int(mem_total)))

	return mem_util

def find_free_host(hostlist, required_gpus, required_mem):

	for host in hostlist:
		gpu_util = query_gpu_utilization(host)

		if len(gpu_util) >= required_gpus:
			queued_gpus = []
			for gpui, gpu in enumerate(gpu_util):
				if (gpu[1]-gpu[0]) >= required_mem:
					queued_gpus.append(str(gpui))
				if len(queued_gpus) == required_gpus:
					return host, queued_gpus

	return None, []

def _async_dispatch(task, hostlist, rm_failed_logs):
	seed(datetime.datetime.now())
	shuffle(hostlist)  # shuffle to reduce risk of querying the same machines multiple times

	idx, command, required_gpus, required_mem = task
	dispatched = False
	available_host = None
	tries_left = MAX_RETRIES

	while not dispatched:
		available_host, gpuids = find_free_host(hostlist, required_gpus, required_mem)
		if available_host is not None:
			dispatched = True

			command = 'CUDA_VISIBLE_DEVICES=%s %s'%(','.join(gpuids), command)

			key = time_stamped()
			logfilepath = './%s-%s.out' % (key, available_host)
			logfile = open(logfilepath, 'w')
			logfile.write(command+'\n\n')
			_print_info('Dispatching command [%d] on host %s on gpus: %s' % (idx, available_host, ','.join(gpuids)))
			try:
				retcode, lastoutput = call(available_host, command, logfile)
			except KeyboardInterrupt as e:
				_print_error('Interrupted command [%d] on host %s on gpus: %s' % (idx, available_host, ','.join(gpuids)))
				return None, None, None
			finally:
				logfile.close()

			if retcode != 0:
				if rm_failed_logs:
					os.remove(logfilepath)

				if tries_left > 0:
					_print_warning('Error while executing command [%d] on host %s on gpus %s. Trying again ...' % (idx, available_host, ','.join(gpuids)))
					_print_warning('Last output line:\n%s' % lastoutput)
					shuffle(hostlist)
					dispatched = False
					tries_left -= 1
				else:
					_print_error('Error while executing command [%d] on host %s on gpus %s. Skipping job:' % (idx, available_host, ','.join(gpuids)))
					_print_error('-------------------------------------')
					_print_error(command)
					_print_error('-------------------------------------')
			else:
				_print_ok('Finished command [%d] on host %s on gpus: %s' % (idx, available_host, ','.join(gpuids)))
			
		if not dispatched:
			time.sleep(1)  # sleep 1 second
	return available_host, gpuids, command


def dispatch(hostlist, commands, required_gpus=1, required_mem=8000, rm_failed_logs=False):
	import multiprocessing as mp

	"""
	hostlist: list of hostnames
	commands: list of strings, as would be written in shell
	required_gpus: integer or list of integers. If list, len(required_gpus) must be equal to len(commands)
	required_mem: in MB. integer or list of integers. Each gpu will be required to have at least this amount of free memory.
	"""

	if type(required_gpus) is list and len(required_gpus) != len(commands):
		raise RuntimeError('Entries in required_gpus list must be equal to entries in commands.')
	if type(required_mem) is list and len(required_mem) != len(commands):
		raise RuntimeError('Entries in required_mem list must be equal to entries in commands.')

	if type(required_gpus) is int:
		required_gpus = [required_gpus]*len(commands)
	if type(required_mem) is int:
		required_mem = [required_mem]*len(commands)

	
	pool = mp.Pool(processes=MAX_PARALLEL_JOBS)

	cmdinds = range(len(commands))
	pool.map_async(partial(_async_dispatch, hostlist=hostlist, rm_failed_logs=rm_failed_logs), zip(cmdinds, commands, required_gpus, required_mem)).get(9999999)

	pool.close()



