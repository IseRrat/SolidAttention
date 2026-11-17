"""Process ownership and termination shared by launchers and experiment tools."""
from contextlib import contextmanager
import os
import signal
import subprocess

HARD_STOP = getattr(signal,"SIGKILL",9)


@contextmanager
def termination_signals():
    interrupted = False
    def interrupt(signum, frame):
        nonlocal interrupted
        if interrupted: return
        interrupted = True
        raise KeyboardInterrupt(f"received signal {signum}")
    previous = signal.signal(signal.SIGTERM,interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM,previous)


def stop_process(process, group=False, timeout=10):
    def send(sig):
        try:
            if group: os.killpg(process.pid,sig)
            elif process.poll() is None:
                if sig == signal.SIGTERM: process.terminate()
                else: process.kill()
        except ProcessLookupError:
            pass  # The child can exit between poll() and signal delivery.
    send(signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        send(HARD_STOP)
        process.wait()
    finally:
        # A launcher can exit before a descendant. Reap the owned session even
        # when the session leader has already exited.
        if group: send(HARD_STOP)


def run_owned(command, *, check=False, timeout=None, **kwargs):
    """Wait for an owned process session and clean it up on any exception."""
    with termination_signals():
        process = subprocess.Popen(command,start_new_session=True,**kwargs)
        try:
            stdout,stderr = process.communicate(timeout=timeout)
        except BaseException as primary:
            try: stop_process(process,group=True)
            except BaseException as cleanup: raise primary from cleanup
            raise
        result = subprocess.CompletedProcess(command,process.returncode,stdout,stderr)
        if check: result.check_returncode()
        return result
