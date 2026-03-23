import os
import timm

def main():
    # Force the cache directories to match your SLURM script
    os.environ["HF_HOME"] = "/scratch/smiyyapuram/medusa/.cache/huggingface"
    os.environ["TORCH_HOME"] = "/scratch/smiyyapuram/medusa/.cache/torch"

    model_name = "resnet18"
    print(f"Downloading weights for {model_name}...")
    
    # This will trigger the download and save it to the cache directories
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,
    )
    
    print("Download complete! The weights are now cached and ready for offline use.")

if __name__ == "__main__":
    main()
