# 实验脚本拥有的进程组在退出、中断和异常时的清理检查。
import os
from pathlib import Path
import signal
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import process_lifetime as lifetime


class ProcessLifetimeTests(unittest.TestCase):
    def test_exit_between_poll_and_signal_is_reaped(self):
        process=mock.Mock(pid=99999999)
        with mock.patch.object(lifetime.os,"killpg",side_effect=ProcessLookupError,create=True):
            lifetime.stop_process(process,group=True)
        process.wait.assert_called_once()

    def test_unresponsive_process_group_is_killed_and_waited(self):
        process=mock.Mock(pid=99999999)
        process.wait.side_effect=[subprocess.TimeoutExpired("synthetic",.01),0]
        with mock.patch.object(lifetime.os,"killpg",create=True) as send:
            lifetime.stop_process(process,group=True,timeout=.01)
        self.assertIn(mock.call(process.pid,lifetime.HARD_STOP),send.call_args_list)
        self.assertEqual(process.wait.call_count,2)

    def test_owned_call_cleans_session_before_propagating_interruption(self):
        process=mock.Mock()
        process.communicate.side_effect=KeyboardInterrupt("cancel synthetic call")
        with mock.patch.object(lifetime.subprocess,"Popen",return_value=process) as spawn, \
                mock.patch.object(lifetime,"stop_process") as stop:
            with self.assertRaisesRegex(KeyboardInterrupt,"cancel synthetic"):
                lifetime.run_owned(["synthetic"],check=True)
        self.assertTrue(spawn.call_args.kwargs["start_new_session"])
        stop.assert_called_once_with(process,group=True)

    def test_termination_handler_restores_owner_and_allows_cleanup(self):
        previous=signal.getsignal(signal.SIGTERM)
        with lifetime.termination_signals():
            handler=signal.getsignal(signal.SIGTERM)
            with self.assertRaises(KeyboardInterrupt): handler(signal.SIGTERM,None)
            self.assertIsNone(handler(signal.SIGTERM,None))
        self.assertIs(signal.getsignal(signal.SIGTERM),previous)

    @unittest.skipUnless(sys.platform.startswith("linux"),"POSIX process-session integration")
    def test_real_synthetic_descendant_stops_with_its_owner(self):
        import selectors
        import time
        code="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(p.pid,flush=True); time.sleep(60)"
        process=subprocess.Popen([sys.executable,"-c",code],stdout=subprocess.PIPE,text=True,start_new_session=True)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout,selectors.EVENT_READ)
                self.assertTrue(selector.select(timeout=5),"synthetic child did not report its PID")
            child=int(process.stdout.readline())
            self.assertEqual(os.getpgid(child),process.pid)
            lifetime.stop_process(process,group=True,timeout=.2)
            deadline=time.monotonic()+2
            state=None
            while time.monotonic()<deadline:
                path=Path(f"/proc/{child}/stat")
                try: state=path.read_text().split(") ",1)[1].split()[0]
                except FileNotFoundError: state=None
                if state in (None,"Z","X"): break
                time.sleep(.01)
            self.assertIn(state,(None,"Z","X"),"synthetic descendant kept executing after cancellation")
        finally:
            if process.poll() is None: lifetime.stop_process(process,group=True,timeout=.2)
            process.stdout.close()


if __name__=="__main__": unittest.main()
