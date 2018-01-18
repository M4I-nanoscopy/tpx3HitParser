import logging
import os
import subprocess
import h5py

logger = logging.getLogger('root')

class io:
    write = None
    read = None

    def open_read(self, file_name):
        f = h5py.File(settings.hits, 'r')

    def open_write(self, file_name, overwrite, ammend):
        if os.path.exists(file_name) and not overwrite:
            logger.error("Output file already exists and --overwrite not specified.")
            raise IOException

        if ammend:
            mode = 'a+'
        else:
            mode = 'w'

        try:
            self.write = h5py.File(file_name, mode)
        except:
            raise

class IOException(Exception):
    pass

# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
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
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION
