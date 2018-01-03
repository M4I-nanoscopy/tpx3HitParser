import signal


# Initializes a multi processing worker and prevents the interupt signal to be handled. This should be handled by the
# parent process.
def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
