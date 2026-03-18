import kagglehub
import shutil
import os

# Define the competition slug
COMPETITION_SLUG = "march-machine-learning-mania-2026"

def download_competition_data():
    print(f"Downloading data for {COMPETITION_SLUG}...")
    try:
        # Download the latest version
        path = kagglehub.competition_download(COMPETITION_SLUG)
        print(f"Data downloaded to: {path}")

        # Target directory in the local workspace
        target_dir = os.path.join(os.getcwd(), "data")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Move files from the cache to the local data directory
        for item in os.listdir(path):
            s = os.path.join(path, item)
            d = os.path.join(target_dir, item)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        
        print(f"Files successfully moved to: {target_dir}")
        return target_dir
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    download_competition_data()
