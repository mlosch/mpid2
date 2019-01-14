import subprocess
import time
import sys
import datetime
try: 
    import queue as queue
except ImportError:
    import Queue as queue
from functools import partial
from random import shuffle, seed
import multiprocessing as mp
import os

MAX_PARALLEL_JOBS = 4

"""
How often does a job tries to rerun a job if it failed
"""
MAX_RETRIES = 3

"""
At below what temperature is a gpu considered unused
"""
MAX_TEMPERATURE = 40

"""
How much time in seconds does a host gets reserved by a job on startup, before allowing other jobs to take the same host
"""
RESERVE_TIME_FOR_JOB_STARTUP = 60


LOG_TARGETS = dict(
	none=None,
	stdout=sys.stdout,
	stderr=sys.stderr,
	file=2,
	file_autorm=3,  # automatically remove the logfile, if the job failed
)


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
				try:
					logfile.write(line)
					logfile.flush()
				except IOError as e:
					print(e)

				output = line
			else:
				output.append(line.strip())
	p.wait()

	rc = p.returncode

	return rc, output


def query_gpu_users(host):
	"""Queries current users of gpus on remote host

	Parameters
	----------
	host : str
		Hostname or address

	Returns
	----------
	list
		List of user names
	"""
	cmd = 'nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs ps -o user'
	retcode, output = remote_exec(host, cmd)

	if retcode != 0:
		raise RuntimeError('Could not query gpu info via nvidia-smi on %s'%host)

	return output[1:]


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

	cmd = 'nvidia-smi --query-gpu="memory.used,memory.total,temperature.gpu" --format=csv,noheader,nounits'
	retcode, output = remote_exec(host, cmd)

	if retcode != 0:
		raise RuntimeError('Could not query gpu info via nvidia-smi on %s'%host)

	num_gpus = len(output)
	mem_util = []
	for gpu in output:
		mem_used, mem_total, temp = gpu.split(', ')
		mem_util.append((int(mem_used), int(mem_total), int(temp)))

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

	conditions_satisfiable = False

	for host in hostlist:
		gpu_util = query_gpu_utilization(host)

		if len(gpu_util) >= required_gpus:
			# check for satisfiability
			req_mem_satisfiable = 0
			for mem_used, mem_total, temp in gpu_util:
				if mem_total > required_mem:
					req_mem_satisfiable += 1
			if req_mem_satisfiable >= required_gpus:
				conditions_satisfiable = True

			queued_gpus = []
			for gpui, gpu in enumerate(gpu_util):
				if gpu[2] > MAX_TEMPERATURE:
					continue
				if (gpu[1]-gpu[0]) >= required_mem:
					queued_gpus.append(str(gpui))
				if len(queued_gpus) == required_gpus:
					return host, queued_gpus

	if not conditions_satisfiable:
		raise RuntimeError('Conditions based on %d required gpus with %dMB each is not satisfiable by any machine at any time.' % (required_gpus, required_mem))

	return None, []


def __interrupt_safe_put(q, obj, timeout=0.1):
	while True:
		try:
			q.put(obj, timeout=timeout)
		except queue.Full:
			continue
		return

def __interrupt_safe_get(q, timeout=0.1):
	while True:
		try:
			obj = q.get(timeout=timeout)
		except queue.Empty:
			continue
		return obj


def _async_dispatch(task, queue_pending, queue_ready, log_target):
	"""Dispatch helper loop. Do not call individually.

	Loops randomly over all hosts and dispatches the given command if a host satisfies the given requirements on memory and number of gpus.

	Parameters
	----------
	task : tuple
		Tuple of length 4 (int, str, int, int), stating the command id, the command string, the number of required gpus and the amount of required free memory in Megabytes.
	queue_pending : multiprocessing.Queue
		Enqueues list of hostnames that have not been evaluated for utilization yet
	queue_ready : multiprocessing.Queue
		Enqueues list of hosts that satisfy gpu requirements and are ready to use
	log_target : str
		One of the keys in LOG_TARGETS 

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

	idx, command = task
	dispatched = False
	available_host = None
	tries_left = MAX_RETRIES

	while not dispatched:
		access_info = None
		while access_info is None:
			access_info = __interrupt_safe_get(queue_ready)
			# if the information is greater than 5 seconds old, put back in pending queue
			if (time.time() - access_info['t']) > 5.0:
				__interrupt_safe_put(queue_pending, (access_info['hostname'], 0))
				access_info = None

		available_host, gpuids = access_info['hostname'], access_info['gpuids']

		if available_host is not None:
			dispatched = True

			host_command = 'export CUDA_VISIBLE_DEVICES=%s; %s'%(','.join(gpuids), command)

			key = time_stamped()
			if log_target.startswith('file'):
				logfilepath = './%s-[%d]-%s.out' % (key, idx, available_host)
				print('Log file at: %s'%logfilepath)
				logfile = open(logfilepath, 'w')
				logfile.write(host_command+'\n\n')
			else:
				logfile = LOG_TARGETS[log_target]

			_print_info('Dispatching command [%d] on host %s on gpus: %s' % (idx, available_host, ','.join(gpuids)))
			t_start = time.time()
			try:
				 # disable this host for 10 seconds to allow enough time to reserve memory
				__interrupt_safe_put(queue_pending, (available_host, t_start + RESERVE_TIME_FOR_JOB_STARTUP))
				retcode, lastoutput = remote_exec(available_host, host_command, logfile)
			except KeyboardInterrupt as e:
				_print_error('Interrupted command [%d] on host %s on gpus: %s' % (idx, available_host, ','.join(gpuids)))
				return None, None, None
			finally:
				t_end = time.time()
				if logfile is not None and logfile is not sys.stdout and logfile is not sys.stderr:
					logfile.close()

			if retcode != 0:

				if log_target == 'file_autorm' and (t_end - t_start) < 10.:
					os.remove(logfilepath)
				if log_target.startswith('file'):
					if os.path.isfile(logfilepath):
						os.rename(logfilepath, logfilepath[:-4]+'_failed.out')


				if tries_left > 0:
					_print_warning('Error while executing command [%d] on host %s on gpus %s. Trying again ...' % (idx, available_host, ','.join(gpuids)))
					_print_warning('Last output line:\n%s' % lastoutput)
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
			print('Command [%d] pending...' % idx)
			# time.sleep(1)  # sleep 1 second
	return available_host, gpuids, host_command


def _utilization_enqueuer(queue_ready, queue_pending, required_gpus, required_mem):
	conditions_satisfiable = False

	# if not conditions_satisfiable:
	# 	raise RuntimeError('Conditions based on %d required gpus with %dMB each is not satisfiable by any machine at any time.' % (required_gpus, required_mem))

	while True:
		host, twait = __interrupt_safe_get(queue_pending)

		if host == '':
			# We're done
			break

		if time.time() < twait:
			# do not evaluate this host yet
			__interrupt_safe_put(queue_pending, (host, twait))
			continue

		timestamp = time.time()
		gpu_util = query_gpu_utilization(host)

		if len(gpu_util) >= required_gpus:
			# check for satisfiability
			req_mem_satisfiable = 0
			for mem_used, mem_total, temp in gpu_util:
				if mem_total > required_mem:
					req_mem_satisfiable += 1
			if req_mem_satisfiable >= required_gpus:
				conditions_satisfiable = True

			queued_gpus = []
			satisfied = False
			for gpui, gpu in enumerate(gpu_util):
				if gpu[2] > MAX_TEMPERATURE:
					continue
				if (gpu[1]-gpu[0]) >= required_mem:
					queued_gpus.append(str(gpui))
				if len(queued_gpus) == required_gpus:
					satisfied = True
					break

			if satisfied:
				try:
					queue_ready.put({'t': timestamp, 'hostname': host, 'gpuids': queued_gpus}, timeout=0.1)
				except queue.Full as e:
					# put back in pending queue, if queue_ready is full (that should not happen)
					print('Unexpected Exception caught: queue.Full')
					queue_pending.put((host, 0), timeout=0.1)
			else:
				# put back and check later in 5 seconds
				__interrupt_safe_put(queue_pending, (host, time.time() + 5))
		else:
			# otherwise, leave the host out of our list entirely. It will never satisfy our requirements
			print('never satisfies: %s'%host)
			pass




def dispatch(hostlist, commands, required_gpus=1, required_mem=8000, log_target='file', rm_failed_logs=False):
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
	log_target : str
		One of the keys in LOG_TARGETS 
	"""

	if type(required_gpus) is list and len(required_gpus) != len(commands):
		raise RuntimeError('Entries in required_gpus list must be equal to entries in commands.')
	if type(required_mem) is list and len(required_mem) != len(commands):
		raise RuntimeError('Entries in required_mem list must be equal to entries in commands.')

	# if type(required_gpus) is int:
	# 	required_gpus = [required_gpus]*len(commands)
	# if type(required_mem) is int:
	# 	required_mem = [required_mem]*len(commands)

	
	pool = mp.Pool(processes=MAX_PARALLEL_JOBS)
	m = mp.Manager()

	# fill queue
	queue_pending = m.Queue(len(hostlist)+1)
	queue_ready = m.Queue(len(hostlist)+1)
	for host in hostlist:
		queue_pending.put((host, 0))

	# start enqueuer
	enqueuer = mp.Process(target=_utilization_enqueuer, args=(queue_ready, queue_pending, required_gpus, required_mem))
	enqueuer.start()

	cmdinds = range(len(commands))
	pool.map_async(
		partial(_async_dispatch, queue_pending=queue_pending, queue_ready=queue_ready, log_target=log_target), 
		zip(cmdinds, commands)
		).get(9999999)

	print('Please wait for processes to finish...')
	pool.close()
	queue_pending.put(('', 0))

	pool.join()
	enqueuer.join()

