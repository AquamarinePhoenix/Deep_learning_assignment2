import os
import random
from PIL import Image
from torch.utils.data import Dataset


def create_mosaic(images, size=224):
    """
    Create a mosaic image from 4 images arranged in a 2x2 grid.
    Each image is resized to size x size, then combined.
    """
    half_size = size // 2
    mosaic = Image.new("RGB", (size, size))
    
    # If fewer than 4 images, repeat or pad
    while len(images) < 4:
        images = images + images[:4 - len(images)]
    
    # Resize and place each image in a quadrant
    positions = [(0, 0), (half_size, 0), (0, half_size), (half_size, half_size)]
    for img, (x, y) in zip(images[:4], positions):
        resized = img.resize((half_size, half_size), Image.Resampling.LANCZOS)
        mosaic.paste(resized, (x, y))
    
    return mosaic


class CaptionDataset(Dataset):
    def __init__(
        self,
        data_dict,
        image_dir,
        processor,
        image_subdir="train",
        image_key="image",
        caption_key="caption_lt",
        caption_en_key="caption_en",
        image_path_key="image_path",
    ):
        self.data = data_dict
        self.image_dir = image_dir
        self.processor = processor
        self.image_subdir = image_subdir
        self.image_key = image_key
        self.caption_key = caption_key
        self.caption_en_key = caption_en_key
        self.image_path_key = image_path_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img_path = item.get(self.image_path_key)
        if img_path:
            # If an explicit image_path is provided, prefer it. If it's not absolute
            # and doesn't exist as given, try joining with the dataset image dir.
            if not os.path.isabs(img_path) and not os.path.exists(img_path):
                candidate = os.path.join(self.image_dir, img_path)
                if os.path.exists(candidate):
                    img_path = candidate
        else:
            img_value = item[self.image_key]
            if os.path.isabs(img_value):
                img_path = img_value
            else:
                img_path = os.path.join(self.image_dir, self.image_subdir, img_value)

        image = Image.open(img_path).convert("RGB")
        caption = item.get(self.caption_key, item.get("caption", ""))

        return {
            "image": image,
            "caption": caption,
            "caption_en": item.get(self.caption_en_key, item.get("caption_en", caption)),
            "image_name": os.path.basename(img_path) if os.path.isabs(img_path) else item.get(self.image_key, os.path.basename(img_path)),
            "image_path": img_path,
        }
        
def collate_fn(batch, apply_mosaic=True, mosaic_probability=0.5):
    """
    Collate batch with optional mosaic augmentation.
    Mosaic combines 4 images into a 2x2 grid for data augmentation.
    """
    images = [x["image"] for x in batch]
    captions = [x["caption"] for x in batch]
    captions_en = [x["caption_en"] for x in batch]
    image_names = [x["image_name"] for x in batch]
    image_paths = [x["image_path"] for x in batch]
    
    # Apply mosaic augmentation with probability
    if apply_mosaic and len(images) >= 2 and random.random() < mosaic_probability:
        # Create mosaic from available images
        mosaic_img = create_mosaic(images)
        # Use first caption for the mosaic (or combine them)
        combined_caption = " + ".join(captions)
        combined_caption_en = " + ".join(captions_en)
        
        return {
            "image": [mosaic_img] + images,
            "caption": [combined_caption] + captions,
            "caption_en": [combined_caption_en] + captions_en,
            "image_name": ["mosaic_" + "_".join(image_names[:4])] + image_names,
            "image_path": ["mosaic"] + image_paths,
        }
    
    return {
        "image": images,
        "caption": captions,
        "caption_en": captions_en,
        "image_name": image_names,
        "image_path": image_paths,
    }