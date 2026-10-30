"""Append-only, hash-chained audit log (tamper-evident)."""
import hashlib
import json
import time
from pathlib import Path


class AuditLog:
    def __init__(self, path: str = "data/audit.log"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._prev = self._last_hash()

    def _last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return "GENESIS"
        last = self.path.read_text().strip().splitlines()[-1]
        return json.loads(last)["hash"]

    def log(self, actor: str, action: str, **details) -> None:
        rec = {"ts": time.time(), "actor": actor, "action": action, "details": details, "prev": self._prev}
        rec["hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        self._prev = rec["hash"]

    def verify(self) -> bool:
        prev = "GENESIS"
        for line in self.path.read_text().splitlines():
            rec = json.loads(line)
            h = rec.pop("hash")
            if rec["prev"] != prev or hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest() != h:
                return False
            prev = h
        return True
