"""launchd job for `cortex daily` on Surya's Mac (macOS's cron; runs a missed time when the Mac wakes).

Why local and not GitHub Actions: the DB, the master resume and Ollama live on the Mac; shipping them to
CI would put personal data in a cloud runner. The plist holds paths only, never secrets: `cortex` reads
.env from the working directory at run time. Output goes to data/daily.log (gitignored data/).
"""
import plistlib
import shutil
import sys
from pathlib import Path

LABEL = "com.cortex.daily"


def plist_dict(repo: Path, cortex_bin: Path, hour: int, minute: int) -> dict:
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("schedule time must be 00:00-23:59")
    log = str(repo / "data" / "daily.log")
    return {"Label": LABEL, "ProgramArguments": [str(cortex_bin), "daily"], "WorkingDirectory": str(repo),
            "StartCalendarInterval": {"Hour": hour, "Minute": minute}, "StandardOutPath": log,
            "StandardErrorPath": log, "RunAtLoad": False, "ProcessType": "Background"}


def render_plist(repo: Path, cortex_bin: Path, hour: int, minute: int) -> bytes:
    return plistlib.dumps(plist_dict(repo, cortex_bin, hour, minute))


def find_cortex_bin() -> Path:
    """The `cortex` next to the running interpreter (the venv's), else whatever is on PATH."""
    here = Path(sys.executable).parent / "cortex"
    if here.exists():
        return here.resolve()
    found = shutil.which("cortex")
    if not found:
        raise FileNotFoundError("can't find the cortex executable: activate the venv and run again")
    return Path(found).resolve()


def install(repo: Path, cortex_bin: Path, hour: int, minute: int, agents_dir: Path | None = None) -> Path:
    agents_dir = agents_dir or Path.home() / "Library" / "LaunchAgents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (repo / "data").mkdir(exist_ok=True)
    path = agents_dir / f"{LABEL}.plist"
    path.write_bytes(render_plist(repo, cortex_bin, hour, minute))
    return path
