import argparse
from pathlib import Path

from PIL import ExifTags, Image


def auto_orient_and_strip(image_path: Path, save_path: Path) -> None:
    img = Image.open(image_path)

    try:
        exif = img._getexif()
        if exif is not None:
            for tag, value in exif.items():
                if ExifTags.TAGS.get(tag) == "Orientation":
                    if value == 3:
                        img = img.rotate(180, expand=True)
                    elif value == 6:
                        img = img.rotate(270, expand=True)
                    elif value == 8:
                        img = img.rotate(90, expand=True)
                    break
    except Exception:
        pass

    # Save the image after applying EXIF orientation and drop EXIF metadata.
    img.save(save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-orient images using EXIF metadata")
    parser.add_argument("--input-dir", required=True, help="Directory containing source images")
    parser.add_argument("--output-dir", required=True, help="Directory to save auto-oriented images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(input_dir.iterdir()):
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            auto_orient_and_strip(
                image_path=image_path,
                save_path=output_dir / image_path.name,
            )


if __name__ == "__main__":
    main()
