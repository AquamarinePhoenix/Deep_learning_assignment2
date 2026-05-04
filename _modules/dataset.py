import os
from PIL import Image
from torch.utils.data import Dataset

class CaptionDataset(Dataset):
    def __init__(self, data_dict, image_dir, processor):
        self.data = data_dict
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img_path = os.path.join(self.image_dir, "train", item["image"])
        image = Image.open(img_path).convert("RGB")

        return {
            "image": image,
            "caption": item["caption_lt"],
            "caption_en": item["caption_en"],
            "image_name": item["image"]
        }
        
def collate_fn(batch):
    return {
        "image": [x["image"] for x in batch],
        "caption": [x["caption"] for x in batch],
        "caption_en": [x["caption_en"] for x in batch],
        "image_name": [x["image_name"] for x in batch],
    }