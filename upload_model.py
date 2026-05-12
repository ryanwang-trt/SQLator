from huggingface_hub import HfApi
from config import MODEL_PATH, HF_MODEL_ID

def upload_model():
    api = HfApi()

    api.create_repo(repo_id=HF_MODEL_ID, repo_type="model", exist_ok=True)
    print(f"Repo '{HF_MODEL_ID}' ready.")

    api.upload_folder(
        folder_path=MODEL_PATH,
        repo_id=HF_MODEL_ID,
        repo_type="model",
    )
    print(f"Uploaded '{MODEL_PATH}' to https://huggingface.co/{HF_MODEL_ID}")

if __name__ == "__main__":
    upload_model()
