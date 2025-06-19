from transformers import BlipProcessor, BlipForConditionalGeneration
import os

def download_blip():
    model_dir = "./local_models/vision/blip_large"
    os.makedirs(model_dir, exist_ok=True)
    
    print("Downloading BLIP large model...")
    
    # This downloads directly without git
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    
    processor.save_pretrained(model_dir)
    model.save_pretrained(model_dir)
    
    print(f"✅ BLIP model saved to: {model_dir}")

if __name__ == "__main__":
    download_blip()

