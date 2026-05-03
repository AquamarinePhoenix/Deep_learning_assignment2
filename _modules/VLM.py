from transformers import BlipProcessor, BlipForConditionalGeneration
import os

def load_VLM(name, device, load_dir=None):
    """Load processor and model. If `load_dir` exists, load from that directory; otherwise load from model name."""
    if load_dir and os.path.exists(load_dir):
        processor = BlipProcessor.from_pretrained(load_dir)
        model = BlipForConditionalGeneration.from_pretrained(load_dir)
    else:
        processor = BlipProcessor.from_pretrained(name)
        model = BlipForConditionalGeneration.from_pretrained(name)

    model.to(device)
    return model, processor, device