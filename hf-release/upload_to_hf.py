import os
import argparse
from huggingface_hub import HfApi
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. If you rely on a .env file, install it via 'pip install python-dotenv'")

def main():
    parser = argparse.ArgumentParser(description="Upload LawUP model to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face Hub repo ID (e.g., your_username/lawup-simplifier-smollm3-3b)")
    parser.add_argument("--folder", type=str, default=".", help="Local folder containing model files to upload")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without actually uploading")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set. Please set it before running the script.")
        print("Example: set HF_TOKEN=your_token (Windows) or export HF_TOKEN=your_token (Linux/macOS)")
        return

    api = HfApi()
    
    try:
        user_info = api.whoami(token=token)
        print(f"Successfully authenticated as: {user_info.get('name')}")
    except Exception as e:
        print(f"Error validating HF_TOKEN: {e}")
        print("Please ensure your token is valid and has the correct permissions (write access).")
        return
    
    print(f"Target repository: {args.repo_id}")
    print(f"Local folder: {args.folder}")
    
    ignore_patterns = [".env", ".env.*", "__pycache__", ".git"]
    
    if args.dry_run:
        print("\n--- DRY RUN: The following files would be uploaded ---")
        for root, dirs, files in os.walk(args.folder):
            # filter out ignored dirs
            dirs[:] = [d for d in dirs if d not in ignore_patterns]
            for file in files:
                if any(file.startswith(pat.strip('*')) for pat in ignore_patterns if pat.startswith('.')):
                    continue
                file_path = os.path.join(root, file)
                print(f" - {file_path}")
        print("--- DRY RUN COMPLETE: Upload skipped. ---")
    else:
        print("Creating repository if it doesn't exist...")
        api.create_repo(repo_id=args.repo_id, exist_ok=True, token=token)
        
        print("Uploading folder...")
        api.upload_folder(
            folder_path=args.folder,
            repo_id=args.repo_id,
            repo_type="model",
            token=token,
            ignore_patterns=ignore_patterns
        )
        print("Upload complete! You can view your model at https://huggingface.co/" + args.repo_id)

if __name__ == "__main__":
    main()
