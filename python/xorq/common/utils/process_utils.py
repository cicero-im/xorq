from subprocess import (
    PIPE,
    Popen,
)
from security import safe_command


def subprocess_run(args, do_decode=False):
    popened = safe_command.run(Popen, args, stdout=PIPE, stderr=PIPE)
    (stdout, stderr) = popened.communicate()
    if do_decode:
        (stdout, stderr) = (el.decode("ascii") for el in popened.communicate())
    return (popened.returncode, stdout, stderr)
