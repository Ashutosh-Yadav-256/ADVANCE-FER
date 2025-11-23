import kagglehub
import shutil
import os

def download_fer2013():
    print("Downloading FER2013 dataset...")
    # Download latest version
    path = kagglehub.dataset_download("msambare/fer2013")
    print("Path to dataset files:", path)
    
    # Define target directory
    target_dir = os.path.join(os.getcwd(), "data", "fer2013")
    
    # Check if we need to move/copy
    if not os.path.exists(target_dir):
        print(f"Moving dataset to {target_dir}...")
        shutil.copytree(path, target_dir)
        print("Dataset moved successfully.")
    else:
        print(f"Target directory {target_dir} already exists. Skipping move.")

if __name__ == "__main__":
    download_fer2013()
