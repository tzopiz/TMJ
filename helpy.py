import os
import json

def rename_files_and_update_annotations(folder_path):
    # Получаем имя папки
    folder_name = os.path.basename(folder_path)

    # Путь к файлу аннотаций
    annotation_file = os.path.join(folder_path, '_annotations.coco.json')

    # Загружаем аннотации
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Получаем список всех файлов в папке
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])

    for i, old_filename in enumerate(files):
        # Новое имя файла
        new_filename = f"{folder_name}_{i:012d}.jpg"

        # Переименовываем файл
        old_filepath = os.path.join(folder_path, old_filename)
        new_filepath = os.path.join(folder_path, new_filename)
        os.rename(old_filepath, new_filepath)

        # Обновляем имена файлов в аннотациях
        for annotation in annotations['images']:
            if annotation['file_name'] == old_filename:
                annotation['file_name'] = new_filename

    # Сохраняем обновленные аннотации
    with open(annotation_file, 'w') as f:
        json.dump(annotations, f, indent=4)
