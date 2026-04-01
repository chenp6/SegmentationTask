"""
1.改 RGB跟COCO json位址
2.python check_caps.py
"""

from pathlib import Path

img_dir = Path("rgb_image") #RGB路徑位置
json_dir = Path("coco_json") #COCO JSON位置

img_names = {p.name.replace("_rgb", ""): p.name for p in img_dir.iterdir()}
json_names = {p.stem: p.name for p in json_dir.iterdir()}

print("=== 不一致（大小寫不同）===")
for key in img_names:
    for jkey in json_names:
        if key.lower() == jkey.lower() and key != jkey:
            print(f"IMG: {img_names[key]}  <--->  JSON: {json_names[jkey]}")

print("\n=== 總計找不到對應 JSON ===")
for key in img_names:
    if key not in json_names:
        print(f"IMG only: {img_names[key]}")

print("\n=== 總計找不到對應 RGB ===")
for key in json_names:
    if key not in img_names:
        print(f"JSON only: {json_names[key]}")
