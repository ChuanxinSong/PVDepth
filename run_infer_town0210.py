import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
PATHS_ENV = SCRIPT_DIR / "paths.env"
INFERENCE_SCRIPT = SCRIPT_DIR / "infer_town0210.py"
BENCHMARK_FILENAMES = (
    "setting_dynamic_fps02_len50.json",
    "setting_dynamic_fps10_len90.json",
    "setting_dynamic_fps20_len110.json",
)


def read_data_root() -> Path:
    for raw_line in PATHS_ENV.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, separator, value = line.partition("=")
        if separator and name.strip() == "data_root":
            data_root = Path(value.strip().strip("'\"")).expanduser()
            return data_root if data_root.is_absolute() else SCRIPT_DIR / data_root

    raise ValueError(f"data_root is not defined in {PATHS_ENV}")


def resolve_json_files() -> List[Path]:
    benchmark_dir = read_data_root() / "test_benchmark"
    json_files = [benchmark_dir / filename for filename in BENCHMARK_FILENAMES]
    missing_files = [path for path in json_files if not path.is_file()]
    if missing_files:
        missing_list = "\n".join(f"  - {path}" for path in missing_files)
        raise FileNotFoundError(f"Required benchmark JSON file(s) not found:\n{missing_list}")

    return json_files


def parse_passthrough_args(argv: Optional[List[str]] = None) -> List[str]:
    parser = argparse.ArgumentParser(
        description="Run Town0210 inference for the configured benchmark JSON files.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--json_path",
        help="Ignored because benchmark JSON files are selected internally.",
    )
    _known_args, passthrough_args = parser.parse_known_args(argv)
    return passthrough_args


def main(argv: Optional[List[str]] = None) -> int:
    passthrough_args = parse_passthrough_args(argv)

    try:
        json_files = resolve_json_files()
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    total = len(json_files)
    print(f"Running {total} Town0210 benchmark tasks.")

    for index, json_path in enumerate(json_files, start=1):
        print(f"[{index}/{total}] {json_path.name}", flush=True)
        command = [
            sys.executable,
            str(INFERENCE_SCRIPT),
            "--json_path",
            str(json_path),
            *passthrough_args,
        ]

        try:
            subprocess.run(command, check=True, text=True)
        except subprocess.CalledProcessError as exc:
            print(f"Error: inference failed for {json_path}", file=sys.stderr)
            print(f"Exit code: {exc.returncode}", file=sys.stderr)
            print(f"Command: {shlex.join(exc.cmd)}", file=sys.stderr)
            return 1

    print(f"Completed all {total} Town0210 benchmark tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
