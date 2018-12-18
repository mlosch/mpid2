import os
import sys
import socket
import datetime
import subprocess

LOGFILE = ''


def time_stamped(fmt='%Y-%m-%d-%H-%M-%S'):
    return datetime.datetime.now().strftime(fmt)

if __name__ == '__main__':

	if not os.path.exists(LOGFILE):
		raise RuntimeError('Set LOGFILE variable first, defining output file for log.')

	# The first argument defines the actual programm/command that will be called
	# You can potentially change this to anything you like, if wanted
	argv = [sys.argv[1]]

	#extract message argument
	msg = None
	for i in range(2, len(sys.argv)):
		if sys.argv[i][0] == '%':
			msg = sys.argv[i][1:]
			if len(msg) == 0:
				raise RuntimeError('Message cannot be empty')
		else:
			argv.append(sys.argv[i])

	if msg is None:
		raise RuntimeError('Missing message. Add anywhere as argument starting with %')

	with open(LOGFILE, 'a') as logfile:
		logline = '{} {} \"{}\" {}\n'.format(time_stamped(), socket.gethostname(), msg, ' '.join(argv))
		logfile.write(logline)

	subprocess.call(argv)
