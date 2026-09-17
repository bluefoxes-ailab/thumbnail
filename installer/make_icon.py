"""
Generates the application icon: the Blue Foxes fox, in the brand blue.

The source fox (frontend/fox.png) is a white silhouette on transparency. A
silhouette recoloured to #1b2fea and left on transparency disappears against
a dark taskbar, which is where this icon spends its life, so the blue is put
behind the fox as a rounded tile and the fox itself stays white. The result
still reads as "the fox, in blue" at 16px, which a dark-on-dark silhouette
does not.

Run from anywhere:  python installer/make_icon.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "frontend" / "fox.png"
OUT = Path(__file__).resolve().parent / "assets" / "fox_blue.ico"
OUT_PNG = Path(__file__).resolve().parent / "assets" / "fox_blue.png"

# Brand blue, straight from the app's own stylesheet (#1b2fea buttons with a
# #026dff border). The tile runs between the two so it doesn't read as a flat
# block at large sizes.
BLUE_DARK = (27, 47, 234)
BLUE_LIGHT = (2, 109, 255)

# Master size. Windows scales the .ico entries down from here; rendering the
# artwork once at 1024 and downsampling with LANCZOS keeps the 16px entry from
# turning into the aliased mush a per-size redraw produces.
MASTER = 1024

# .ico sizes Windows actually asks for: Explorer tiles, the taskbar, the
# title bar, and the 16px tray/menu entry.
ICO_SIZES = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]


def rounded_tile(size: int) -> Image.Image:
    """Blue rounded square with a vertical gradient, as the icon's ground."""
    gradient = Image.new("RGB", (1, size))
    for y in range(size):
        t = y / max(size - 1, 1)
        gradient.putpixel(
            (0, y),
            tuple(round(BLUE_LIGHT[c] + (BLUE_DARK[c] - BLUE_LIGHT[c]) * t) for c in range(3)),
        )
    gradient = gradient.resize((size, size), Image.NEAREST)

    # Windows 11's own icon geometry: a squircle-ish radius of ~22% of the
    # tile, with a small inset so the corners aren't clipped by the frame.
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=round(size * 0.22), fill=255
    )

    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    tile.paste(gradient, (0, 0), mask)
    return tile


def white_fox(box: int) -> Image.Image:
    """The source silhouette, forced to opaque white, fitted into `box`."""
    fox = Image.open(SRC).convert("RGBA")

    # The source is drawn in near-white already, but its edges carry the
    # colour of whatever it was exported over. Only the alpha channel is
    # meaningful, so the RGB is discarded and rebuilt as pure white — this is
    # what keeps the fox from picking up a grey fringe against the blue.
    alpha = fox.split()[3]
    fox = Image.merge("RGBA", (
        Image.new("L", fox.size, 255),
        Image.new("L", fox.size, 255),
        Image.new("L", fox.size, 255),
        alpha,
    ))

    # Trim to the silhouette's real ink before scaling, so the padding baked
    # into the source PNG doesn't shrink the fox inside the tile.
    bbox = alpha.getbbox()
    if bbox:
        fox = fox.crop(bbox)

    scale = min(box / fox.width, box / fox.height)
    return fox.resize((max(1, round(fox.width * scale)), max(1, round(fox.height * scale))), Image.LANCZOS)


def build_master() -> Image.Image:
    icon = rounded_tile(MASTER)

    # 62% of the tile: enough margin that the fox never touches the rounded
    # corners at 16px, where the tile edge and the ink merge visually.
    fox = white_fox(round(MASTER * 0.62))

    x = (MASTER - fox.width) // 2
    y = (MASTER - fox.height) // 2

    # A soft drop shadow keeps the white fox from floating flat on the blue at
    # large sizes. It is far too subtle to survive the 16px downsample, which
    # is fine — at that size the silhouette alone carries the shape.
    shadow = Image.new("RGBA", icon.size, (0, 0, 0, 0))
    shadow.paste((0, 0, 0, 90), (x, y + round(MASTER * 0.012)), fox.split()[3])
    shadow = shadow.filter(ImageFilter.GaussianBlur(MASTER * 0.012))
    icon = Image.alpha_composite(icon, shadow)

    icon.paste(fox, (x, y), fox)
    return icon


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"source fox not found: {SRC}")

    master = build_master()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    master.resize((512, 512), Image.LANCZOS).save(OUT_PNG)
    master.save(OUT, sizes=ICO_SIZES)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes) and {OUT_PNG}")


if __name__ == "__main__":
    main()
