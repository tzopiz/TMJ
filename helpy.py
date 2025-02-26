import os
import json

def rename_files_and_update_annotations(folder_path):
    annotation_file = os.path.join(folder_path, "_annotations.coco.json")
    folder_name = os.path.basename(os.path.normpath(folder_path))

    with open(annotation_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}

    for img in coco_data["images"]:
        img_id = img["id"]
        old_name = img["file_name"]
        new_name = f"{folder_name}_{img_id:012d}.jpg"
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)

        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            img["file_name"] = new_name
            print(f"{old_name} -> {new_name}")
        else:
            print(f"Файл {old_name} не найден")

    with open(annotation_file, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=4, ensure_ascii=False)

