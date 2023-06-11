import signal
import os
import subprocess


# Initializes a multi processing worker and prevents the interupt signal to be handled. This should be handled by the
# parent process.
def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def minimal_ext_cmd(cmd):
    # construct minimal environment
    env = {}
    for k in ['SYSTEMROOT', 'PATH']:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    # LANGUAGE is used on win32
    env['LANGUAGE'] = 'C'
    env['LANG'] = 'C'
    env['LC_ALL'] = 'C'
    cwd = os.path.dirname(os.path.realpath(__file__))
    FNULL = open(os.devnull, 'w')
    o = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env, cwd=cwd, stderr=FNULL).communicate()[0]
    return o
