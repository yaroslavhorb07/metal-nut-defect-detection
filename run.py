"""Start the nut defect detector and open it in the browser.

Run this file (python run.py, or the Run button in your editor).
Press Ctrl+C, or stop the program, to shut the server down.
"""
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", "8000"))
URL = f"http://{HOST}:{PORT}"


def venv_python():
    for p in (ROOT / ".venv" / "bin" / "python", ROOT / ".venv" / "Scripts" / "python.exe"):
        if p.exists():
            return p
    return None


def ensure_venv():
    # the editor may start this with a Python that lacks the project's packages
    venv = venv_python()
    if venv and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        sys.exit(subprocess.call([str(venv), str(ROOT / "run.py"), *sys.argv[1:]]))
    if venv is None:
        try:
            import uvicorn, keras  # noqa: F401
        except ImportError:
            sys.exit("No .venv found and packages are missing. See README.md, 'Instructions to build'.")


def already_running():
    try:
        urllib.request.urlopen(URL + "/health", timeout=1)
        return True
    except Exception:
        return False


def open_when_ready():
    for _ in range(120):
        try:
            urllib.request.urlopen(URL + "/health", timeout=1)
            print(f"\nReady: {URL}  (press Ctrl+C to stop)\n")
            webbrowser.open(URL)
            return
        except Exception:
            time.sleep(0.5)
    print("Server did not start in time, open", URL, "manually.")


def main():
    ensure_venv()
    if already_running():
        print(f"The server is already running, opening {URL}")
        webbrowser.open(URL)
        return
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    import uvicorn

    print(f"Loading the model and starting the server at {URL} ...")
    threading.Thread(target=open_when_ready, daemon=True).start()
    try:
        uvicorn.run("nutdetect.api:app", host=HOST, port=PORT)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
