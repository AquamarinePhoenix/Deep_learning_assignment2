from transformers import BlipProcessor, BlipForConditionalGeneration

def load_VLM(name, device):
    processor = BlipProcessor.from_pretrained(name)
    model = BlipForConditionalGeneration.from_pretrained(name)
    device = device
    model.to(device)
    
    return model, processor, device