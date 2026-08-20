#!/usr/bin/env python3
"""Generates assets/uv_grid.png, the M4 texture-path test image.

Authored here rather than vendored so the repository carries no third-party
image and the asset is reproducible from source. The pattern is chosen to make
UV bugs visible rather than to look good:

  * 8x8 checkerboard of distinct hues -- a swapped or scaled UV shifts colours.
  * A red block at UV (0,0) and a green block at (1,0), so a flipped V axis is
    obvious instead of merely suspicious.
  * One-pixel grid lines, which alias into visible shimmer if mip selection or
    the sampler's LOD range is wrong.
"""

import struct
import zlib
from pathlib import Path

SIZE = 512
CELLS = 8
CELL = SIZE // CELLS

# Distinct, evenly spread hues; index by (cell_x + cell_y) so neighbours differ.
PALETTE = [
    (219, 68, 55), (244, 180, 0), (15, 157, 88), (66, 133, 244),
    (171, 71, 188), (0, 172, 193), (255, 112, 67), (158, 157, 36),
]
DARK = (38, 40, 46)
LINE = (245, 245, 248)
ORIGIN_MARK = (255, 0, 0)     # UV (0,0) -- top-left in glTF and Vulkan alike
U_AXIS_MARK = (0, 255, 0)     # UV (1,0) -- top-right


def pixel(x: int, y: int) -> tuple[int, int, int]:
    cell_x, cell_y = x // CELL, y // CELL

    if cell_x == 0 and cell_y == 0:
        return ORIGIN_MARK
    if cell_x == CELLS - 1 and cell_y == 0:
        return U_AXIS_MARK

    # Grid lines sit on cell boundaries, one pixel wide.
    if x % CELL == 0 or y % CELL == 0:
        return LINE

    if (cell_x + cell_y) % 2 == 0:
        return DARK
    return PALETTE[(cell_x * 3 + cell_y * 5) % len(PALETTE)]


def main() -> None:
    raw = bytearray()
    for y in range(SIZE):
        raw.append(0)  # PNG filter type 0 (None) for this scanline
        for x in range(SIZE):
            raw.extend(pixel(x, y))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", SIZE, SIZE, 8, 2, 0, 0, 0)  # 8-bit RGB
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", header)
           + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
           + chunk(b"IEND", b""))

    out = Path(__file__).resolve().parent.parent / "assets" / "uv_grid.png"
    out.write_bytes(png)
    print(f"wrote {out} ({len(png)} bytes, {SIZE}x{SIZE})")


if __name__ == "__main__":
    main()
