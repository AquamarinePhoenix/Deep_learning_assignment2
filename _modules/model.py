import os
import torch as th
from PIL import Image
import _modules.config as cfg
from _modules.write import write_to_file
from torch.utils.data import DataLoader
from _modules.dataset import CaptionDataset, collate_fn

def train(model, processor, optimizer, data_dict, device, results_file):
    from torch.utils.data import DataLoader
    from _modules.dataset import CaptionDataset

    model.to(device)

    dataset = CaptionDataset(data_dict, cfg.IMAGE_DIR, processor)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    for epoch in range(cfg.EPOCHS):
        model.train()

        total_loss = 0
        count = 0

        write_to_file(results_file, f"\nEpoch {epoch + 1}")

        for batch in loader:
            images = batch["image"]
            captions = batch["caption"]
            image_names = batch["image_name"]

            inputs = processor(
                images=images,
                text=captions,
                return_tensors="pt",
                padding=True
            ).to(device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            count += 1

            # log first epoch samples
            if epoch == 0:
                for i in range(len(image_names)):
                    write_to_file(
                        results_file,
                        f"{i + 1}. {image_names[i]} | {captions[i]}"
                    )

        avg_loss = total_loss / count if count > 0 else 0
        write_to_file(results_file, f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

    model.save_pretrained(cfg.MODEL_SAVE_DIR)
    processor.save_pretrained(cfg.MODEL_SAVE_DIR)

    return model

def evaluate(model, processor, data_dict, device, results_file):
    model.to(device)
    model.eval()

    for item in data_dict:
        img_name = item["image"]
        test_path = cfg.IMAGE_DIR + "test/" + img_name

        if not os.path.exists(test_path):
            continue

        image = Image.open(test_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with th.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=5,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
                temperature=1.0
            )

        pred_lt = processor.decode(out[0], skip_special_tokens=True)

        write_to_file(results_file, f"\nTEST {img_name}")
        write_to_file(results_file, f"Pred_LT: {pred_lt}")
        write_to_file(results_file, f"Real_LT: {item['caption_lt']}")
        write_to_file(results_file, f"Real_EN: {item['caption_en']}")
    return model