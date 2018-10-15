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


def remote_exec(host, command, logfile=None):
	"""Calls a command on a remote host via ssh

	Parameters
	----------
	host : str
		Hostname or address
	command : str
		Command to execute on host
	logfile : filehandle
		If set, all outputs from remotely executed command will be written to this file handle

	Returns
	----------
	int
		Return code of executed command on remote host
	list
		List of command output strings
	"""
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
	"""Queries gpu utilization on remote host

	Parameters
	----------
	host : str
		Hostname or address

	Returns
	----------
	list
		List of integer pairs (memory used, total memory). The length of the list equals the number of gpus on the host
	"""

	cmd = 'nvidia-smi --query-gpu="memory.used,memory.total" --format=csv,noheader,nounits'
	retcode, output = remote_exec(host, cmd)

	if retcode != 0:
		raise RuntimeError('Could not query gpu info via nvidia-smi')

	num_gpus = len(output)
	mem_util = []
	for gpu in output:
		mem_used, mem_total = gpu.split(', ')
		mem_util.append((int(mem_used), int(mem_total)))

	return mem_util


def find_free_host(hostlist, required_gpus, required_mem):
	"""Returns the first host, that matches the required number of gpus and memory

	Parameters
	----------
	hostlist : list
		List of hostnames or addresses
	required_gpus : int
		Number of required gpus
	required_mem : int
		Amount of free memory required per gpu in Megabyte

	Returns
	----------
	str
		Hostname that matches the conditions
	list
		GPU utilizations of the host
	"""

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
	"""Dispatch helper loop. Do not call individually.

	Loops randomly over all hosts and dispatches the given command if a host satisfies the given requirements on memory and number of gpus.

	Parameters
	----------
	task : tuple
		Tuple of length 4 (int, str, int, int), stating the command id, the command string, the number of required gpus and the amount of required free memory in Megabytes.
	hostlist : list
		List of hostnames or addresses
	rm_failed_logs : bool
		Set to true, if created command output logs should be deleted automatically if the remote job failed.

	Returns
	----------
	str
		Hostname, the command was dispatched to
	tuple
		IDs of the gpus used on the host
	str
		A copy of the command executed on the host, including the set CUDA_VISIBLE_DEVICES environment variable.
	"""

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

			command = 'export CUDA_VISIBLE_DEVICES=%s; %s'%(','.join(gpuids), command)

			key = time_stamped()
			logfilepath = './%s-%s.out' % (key, available_host)
			logfile = open(logfilepath, 'w')
			logfile.write(command+'\n\n')
			_print_info('Dispatching command [%d] on host %s on gpus: %s' % (idx, available_host, ','.join(gpuids)))
			try:
				retcode, lastoutput = remote_exec(available_host, command, logfile)
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
	"""Main dispatcher method.

	Arguments
	----------
	hostlist : list 
		List of hostnames or addresses
	commands : list
		List of command strings, as would be written in shell. Ensure the correct working directory by prepending a `cd ~/workdir/...;` if necessary.
	required_gpus : int
		Integer or list of integers defining the minimum number of required gpus on a single host. If list, len(required_gpus) must be equal to len(commands)
	required_mem : int
		In Megabytes. Integer or list of integers, defining the minimum amount of free memory required per gpu on a single host.
	"""

	import multiprocessing as mp

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

