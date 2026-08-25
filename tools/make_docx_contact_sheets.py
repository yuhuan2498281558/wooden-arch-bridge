from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages = sorted(args.input_dir.glob("page-*.png"))
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    label_font = ImageFont.truetype(str(font_path), 24) if font_path.exists() else ImageFont.load_default()
    for offset in range(0, len(pages), 6):
        group = pages[offset : offset + 6]
        thumb_w, thumb_h, margin, label_h = 510, 660, 28, 36
        canvas = Image.new("RGB", (margin * 4 + thumb_w * 3, margin * 3 + (thumb_h + label_h) * 2), "#D8DEE5")
        draw = ImageDraw.Draw(canvas)
        for index, page in enumerate(group):
            img = Image.open(page).convert("RGB")
            img.thumbnail((thumb_w, thumb_h))
            col, row = index % 3, index // 3
            x = margin + col * (thumb_w + margin)
            y = margin + row * (thumb_h + label_h + margin)
            holder = Image.new("RGB", (thumb_w, thumb_h), "white")
            holder.paste(img, ((thumb_w - img.width) // 2, (thumb_h - img.height) // 2))
            canvas.paste(holder, (x, y))
            label = page.stem.replace("page-", "第 ") + " 页"
            draw.text((x + 4, y + thumb_h + 4), label, font=label_font, fill="#22313F")
        canvas.save(args.output_dir / f"contact-{offset // 6 + 1:02d}.png")
    print(f"pages={len(pages)} sheets={(len(pages) + 5) // 6}")


if __name__ == "__main__":
    main()

