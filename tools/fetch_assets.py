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

# The second source. Poly Haven publishes scanned props as CC0, which Khronos
# has nothing equivalent to -- its library is test and showcase assets, not
# scenery. Same stdlib, a different shape: an API call lists a .gltf plus the
# files it references, rather than a directory to enumerate.
USER_AGENT = "sage-fetch-assets"

POLYHAVEN_API = "https://api.polyhaven.com/files"
# 2k rather than 4k or 8k. The maps are all sage reads -- base colour, normal,
# and occlusion-roughness-metallic packed as `arm` -- and 4k would quadruple a
# download already measured in tens of megabytes for no visible gain at the
# resolutions these are captured at.
POLYHAVEN_RESOLUTION = "2k"

# Local directory -> upstream model name. The local names are snake_case to
# match the rest of the tree; upstream uses PascalCase.
MODELS = {
    "lantern": "Lantern",
    "flight_helmet": "FlightHelmet",
    "damaged_helmet": "DamagedHelmet",
    # The v1.0 hero scene. Chess pieces on a board give what a single model
    # cannot: many objects casting shadows onto a surface that is part of the
    # same file, so the shadow pass has something to land on without a
    # procedural ground plane.
    "chess": "ABeautifulGame",
    # The architectural classic, and the one asset here big enough to show what
    # a single shadow map fitted to the whole scene costs: 9.5 MB of geometry
    # across 67 textures.
    "sponza": "Sponza",
    # DamagedHelmet's better-behaved cousin: same idea, but it ships TANGENT, so
    # its normal mapping actually shades correctly.
    "sci_fi_helmet": "SciFiHelmet",
    # Leather, brass and wood over one key light: no extension sage does not
    # implement, and nothing that depends on a reflection it cannot supply.
    "antique_camera": "AntiqueCamera",
}

# Local directory -> Poly Haven slug. Solid geometry only, deliberately: sage
# does no alpha testing -- the mesh shader has no discard and the loader ignores
# alphaMode -- so anything built from alpha-cut leaf cards would draw its
# foliage as opaque rectangles. Rocks, cliffs and stumps have no such problem.
POLYHAVEN_MODELS = {
    "stump": "tree_stump_01",
    "boulder": "boulder_01",
    "mossy_rocks": "rock_moss_set_01",
    "desert_boulder": "namaqualand_boulder_04",
}

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def fetch(url: str) -> bytes:
    # A User-Agent is not decoration: Poly Haven answers urllib's default with
    # 403, and GitHub's API asks for one. Sending it on every request keeps the
    # two sources on one code path.
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
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


def download_polyhaven(local_name: str, slug: str, force: bool) -> int:
    """Fetches one Poly Haven model and the files its glTF references."""
    target = ASSETS / local_name
    target.mkdir(parents=True, exist_ok=True)

    files = json.loads(fetch(f"{POLYHAVEN_API}/{slug}"))
    entry = files["gltf"][POLYHAVEN_RESOLUTION]["gltf"]

    # The glTF itself, then everything it names. `include` keys are paths
    # relative to the glTF, textures/ among them, so directories are created as
    # they are met rather than assumed flat.
    wanted = {f"{slug}_{POLYHAVEN_RESOLUTION}.gltf": entry["url"]}
    for relative, info in entry.get("include", {}).items():
        wanted[relative] = info["url"]

    written = 0
    for relative, url in wanted.items():
        path = target / relative
        if path.exists() and not force:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch(url))
        print(f"  {relative}")
        written += 1

    # Poly Haven ships no per-asset licence file, so the terms are written here
    # rather than left to be looked up.
    licence = target / "LICENSE.md"
    if not licence.exists() or force:
        licence.write_text(
            f"{slug} from Poly Haven (https://polyhaven.com/a/{slug}).\n"
            "Licensed CC0: public domain, no attribution required.\n")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", choices=[*MODELS, *POLYHAVEN_MODELS, []],
                        default=[], help="models to fetch (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="re-download files that already exist")
    parser.add_argument("--list", action="store_true", help="list models and exit")
    args = parser.parse_args()

    if args.list:
        for local, upstream in MODELS.items():
            print(f"{local:<16} {RAW}/{upstream}/glTF")
        for local, slug in POLYHAVEN_MODELS.items():
            print(f"{local:<16} https://polyhaven.com/a/{slug}  (CC0)")
        return 0

    wanted = args.models or [*MODELS, *POLYHAVEN_MODELS]
    for local in wanted:
        print(f"Fetching {local}...")
        try:
            if local in POLYHAVEN_MODELS:
                download_polyhaven(local, POLYHAVEN_MODELS[local], args.force)
            else:
                download_model(local, MODELS[local], args.force)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as error:
            # Reported per model rather than raised: one unreachable model
            # should not cost the others already on disk.
            print(f"{local}: FAILED ({error})", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
