# https://stackoverflow.com/questions/49663124/cause-python-to-exit-if-any-thread-has-an-exception

import os
import threading
import traceback

original_init = threading.Thread.__init__
threadexception = None


def patched_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    original_run = self.run

    def patched_run(*args, **kw):
        try:
            original_run(*args, **kw)
        except SystemExit:
            os._exit(1)
        except BaseException:
            traceback.print_exc()
            os._exit(1)

    self.run = patched_run


def install():
    threading.Thread.__init__ = patched_init
