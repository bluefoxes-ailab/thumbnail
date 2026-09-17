"""
Runs the backend for development, on port 8000.

`start_app.bat` is the normal way in and picks the interpreter for you. This
file is what that runs, and what anybody typing `python run_backend.py`
reaches directly — which is why it says something useful when the interpreter
they typed it into is the wrong one.
"""

import os
import subprocess
import sys

os.chdir(os.path.join(os.path.dirname(__file__), "backend"))
sys.path.insert(0, os.getcwd())

try:
    from api import app
    import uvicorn
except ModuleNotFoundError as missing:
    # A machine can carry three Pythons — a store stub, a plain CPython, the
    # one the dependencies were actually installed into — and Windows resolves
    # a bare `python` against the MACHINE path before the user path, so
    # installing an unrelated Python quietly moves this script onto an
    # interpreter that has none of what it needs.
    #
    # The traceback that produces names the module and not the cause: it reads
    # as a broken checkout ("No module named 'cv2'" out of a file nobody
    # edited), and the obvious next move is to reinstall the requirements —
    # into the same wrong interpreter. So the interpreter is named here, where
    # the failure happens, rather than left to be worked out.
    print(f"{type(missing).__name__}: {missing}\n", file=sys.stderr)
    print(f"This is running on {sys.executable}", file=sys.stderr)
    print("(Python " + sys.version.split()[0] + "), which does not have the "
          "backend's dependencies.\n", file=sys.stderr)
    print("Either install them into it:\n", file=sys.stderr)
    print(f'    "{sys.executable}" -m pip install -r ../requirements.txt\n', file=sys.stderr)
    print("...or start the app with start_app.bat, which picks an interpreter "
          "that already has them.", file=sys.stderr)
    sys.exit(1)

uvicorn.run(app, host="0.0.0.0", port=8000)
