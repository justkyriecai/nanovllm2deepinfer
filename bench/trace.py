import json
from dataclasses import asdict


class TraceWriter:
    """engine.trace callback: one JSON line per engine step."""

    def __init__(self, path):
        self.f = open(path, "w")
        self.step = 0

    def __call__(self, record):
        self.f.write(json.dumps({"step": self.step, **asdict(record)}) + "\n")
        self.step += 1

    def close(self):
        self.f.close()


def read_trace(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]
