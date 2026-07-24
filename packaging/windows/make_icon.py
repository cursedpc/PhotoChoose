from __future__ import annotations

from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_ROOT / "src" / "photochoose" / "branding" / "photochoose-icon.png"
OUTPUT = Path(__file__).with_name("photochoose.ico")
ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def main() -> None:
    with Image.open(SOURCE) as source:
        image = source.convert("RGBA")
        image.save(
            OUTPUT,
            format="ICO",
            sizes=[(size, size) for size in ICON_SIZES],
            bitmap_format="png",
        )
    print("Wrote {0}".format(OUTPUT))


if __name__ == "__main__":
    main()
