from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "splits" / "near_duplicates.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "audit" / "near_duplicates"

IMAGE_SIZE = (500, 500)
HEADER_HEIGHT = 100
FOOTER_HEIGHT = 60
GAP = 20


def load_and_resize(filepath):
    with Image.open(filepath) as image:
        image = image.convert("RGB")
        image.thumbnail(IMAGE_SIZE)

        canvas = Image.new("RGB", IMAGE_SIZE, "black")

        x = (IMAGE_SIZE[0] - image.width) // 2
        y = (IMAGE_SIZE[1] - image.height) // 2

        canvas.paste(image, (x, y))

    return canvas


def draw_centered_text(draw, text, center_x, y, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]

    draw.text(
        (center_x - width // 2, y),
        text,
        fill="white",
        font=font,
    )


def main():
    df = pd.read_csv(CSV_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    font = ImageFont.load_default()

    counters = {}

    for _, row in df.iterrows():
        distance = int(row["distance"])

        distance_dir = OUTPUT_DIR / f"distance_{distance}"
        distance_dir.mkdir(parents=True, exist_ok=True)

        counters.setdefault(distance, 0)
        counters[distance] += 1

        image_1_path = PROJECT_ROOT / row["filepath_1"]
        image_2_path = PROJECT_ROOT / row["filepath_2"]

        image_1 = load_and_resize(image_1_path)
        image_2 = load_and_resize(image_2_path)

        canvas_width = IMAGE_SIZE[0] * 2 + GAP
        canvas_height = HEADER_HEIGHT + IMAGE_SIZE[1] + FOOTER_HEIGHT

        canvas = Image.new(
            "RGB",
            (canvas_width, canvas_height),
            "black",
        )

        canvas.paste(
            image_1,
            (0, HEADER_HEIGHT),
        )

        canvas.paste(
            image_2,
            (IMAGE_SIZE[0] + GAP, HEADER_HEIGHT),
        )

        draw = ImageDraw.Draw(canvas)

        left_center = IMAGE_SIZE[0] // 2
        right_center = IMAGE_SIZE[0] + GAP + IMAGE_SIZE[0] // 2

        draw_centered_text(
            draw,
            f"{row['split_1'].upper()} | {row['group_id_1']}",
            left_center,
            20,
            font,
        )

        draw_centered_text(
            draw,
            Path(row["filepath_1"]).name,
            left_center,
            50,
            font,
        )

        draw_centered_text(
            draw,
            f"{row['split_2'].upper()} | {row['group_id_2']}",
            right_center,
            20,
            font,
        )

        draw_centered_text(
            draw,
            Path(row["filepath_2"]).name,
            right_center,
            50,
            font,
        )

        draw_centered_text(
            draw,
            f"pHash Hamming Distance = {distance}",
            canvas_width // 2,
            HEADER_HEIGHT + IMAGE_SIZE[1] + 20,
            font,
        )

        output_path = (
            distance_dir
            / f"pair_{counters[distance]:03d}.png"
        )

        canvas.save(output_path)

    print("\nNear duplicate visualization complete.")

    for distance in sorted(counters):
        print(
            f"distance {distance}: "
            f"{counters[distance]} pairs"
        )

    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()