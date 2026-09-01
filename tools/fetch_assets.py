#!/usr/bin/env python3
"""Downloads the glTF test models into assets/.

The models were vendored until M6 and are now fetched instead: they had grown
to 56 MB, which is a clone every contributor pays for so that the renderer has
something to open. This script buys that back -- run it once and the models are
there, skip it and the app starts on an empty scene, which it is built to do.

The separate-file (`glTF/`) variant is downloaded rather than the binary
(`glTF-Binary/`) one on purpose: sage reads images by URI, and a .glb keeps
them in a buffer view, which the loader reports and falls back to white for.

Stdlib only, so this stays runnable on a fresh clone with no pip install:

    python3 tools/fetch_assets.py                 # everything
    python3 tools/fetch_assets.py lantern         # one model
    python3 tools/fetch_assets.py --list
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "KhronosGroup/glTF-Sample-Assets"
CONTENTS = f"https://api.github.com/repos/{REPO}/contents/Models"
RAW = f"https://raw.githubusercontent.com/{REPO}/main/Models"

# Local directory -> upstream model name. The local names are snake_case to
# match the rest of the tree; upstream uses PascalCase.
MODELS = {
    "lantern": "Lantern",
    "flight_helmet": "FlightHelmet",
    "damaged_helmet": "DamagedHelmet",
}

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        return response.read()


def download_model(local_name: str, upstream_name: str, force: bool) -> int:
    target = ASSETS / local_name
    target.mkdir(parents=True, exist_ok=True)

    listing = json.loads(fetch(f"{CONTENTS}/{upstream_name}/glTF"))
    # LICENSE.md sits one level up, beside the variant directories. Fetched
    # alongside the model so the terms travel with the files.
    entries = [(item["name"], item["download_url"]) for item in listing]
    entries.append(("LICENSE.md", f"{RAW}/{upstream_name}/LICENSE.md"))

    written = 0
    for name, url in entries:
        path = target / name
        if path.exists() and not force:
            continue
        print(f"  {name}", flush=True)
        path.write_bytes(fetch(url))
        written += 1

    total = sum(f.stat().st_size for f in target.iterdir() if f.is_file())
    verb = "downloaded" if written else "already present"
    print(f"{local_name}: {verb} ({total / 1e6:.1f} MB, {len(entries)} files)")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", choices=[*MODELS, []], default=[],
                        help="models to fetch (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="re-download files that already exist")
    parser.add_argument("--list", action="store_true", help="list models and exit")
    args = parser.parse_args()

    if args.list:
        for local, upstream in MODELS.items():
            print(f"{local:<16} {RAW}/{upstream}/glTF")
        return 0

    wanted = args.models or list(MODELS)
    for local in wanted:
        print(f"Fetching {local}...")
        try:
            download_model(local, MODELS[local], args.force)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as error:
            # Reported per model rather than raised: one unreachable model
            # should not cost the others already on disk.
            print(f"{local}: FAILED ({error})", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
