from tqdm import tqdm
import cv2
import numpy as np
import os

class MaskCreator:
    def __init__(self, coco, image_shapes):
        self.coco = coco
        self.image_shapes = image_shapes

    def _create_mask(self, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        height, width = self.image_shapes[image_id]
        unique_categories = list(set(ann['category_id'] for ann in anns))

        masks = np.zeros((height, width, len(unique_categories)), dtype=np.uint8)
        category_to_index = {cat_id: i for i, cat_id in enumerate(unique_categories)}

        for ann in anns:
            category_id = ann['category_id']
            mask = self.coco.annToMask(ann)
            idx = category_to_index[category_id]
            masks[:, :, idx] += mask

        return masks, unique_categories

    def save_all_masks(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        image_ids = self.coco.getImgIds()

        for image_id in tqdm(image_ids, desc="Processing images"):
            if image_id not in self.image_shapes:
                print(f"Warning: No shape info for image {image_id}")
                continue

            masks, categories = self._create_mask(image_id)
            mask_path = os.path.join(save_dir, f"{image_id}.npy")
            np.save(mask_path, masks)
            np.save(mask_path.replace(".npy", "_classes.npy"), np.array(categories))

        print(f"All masks saved in {save_dir}")

    def save_all_masks_as_images(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        image_ids = self.coco.getImgIds()

        for image_id in tqdm(image_ids, desc="Processing images"):
            if image_id not in self.image_shapes:
                print(f"Warning: No shape info for image {image_id}")
                continue

            masks, categories = self._create_mask(image_id)

            for i, category_id in enumerate(categories):
                mask_path = os.path.join(save_dir, f"{image_id}_class{category_id}.png")
                mask = (masks[:, :, i] * 255).astype(np.uint8)
                cv2.imwrite(mask_path, mask)

        print(f"All masks saved as images in {save_dir}")

    def save_all_masks_as_colored_images(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        image_ids = self.coco.getImgIds()

        unique_categories = list(set(ann['category_id'] for ann in self.coco.loadAnns(self.coco.getAnnIds())))
        category_colors = {cat_id: np.random.randint(50, 255, size=(3,), dtype=np.uint8) for cat_id in unique_categories}

        for image_id in tqdm(image_ids, desc="Processing images"):
            if image_id not in self.image_shapes:
                print(f"Warning: No shape info for image {image_id}")
                continue

            masks, categories = self._create_mask(image_id)
            height, width = self.image_shapes[image_id]

            colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

            for i, category_id in enumerate(categories):
                color = category_colors[category_id]
                mask = masks[:, :, i]
                for c in range(3):
                    colored_mask[:, :, c] += (mask * color[c]).astype(np.uint8)

            mask_path = os.path.join(save_dir, f"{image_id}_colored.png")
            cv2.imwrite(mask_path, colored_mask)

        print(f"All colored masks saved in {save_dir}")
