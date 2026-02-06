#!/usr/bin/env python3
"""
DVC Setup Script for Fraud Detection MLOps POC
Initializes DVC and configures local remote storage for dataset versioning
"""

import os
import sys
import subprocess
import shutil

def run_command(cmd, cwd=None, use_dvc_module=False):
    """Run shell command and return output"""
    try:
        # If it's a DVC command, use python -m dvc
        if use_dvc_module or cmd.startswith('dvc '):
            cmd = cmd.replace('dvc ', f'{sys.executable} -m dvc ', 1)
        
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {cmd}")
        print(f"        {e.stderr}")
        return None

def check_dvc_installed():
    """Check if DVC is installed"""
    try:
        result = subprocess.run(
            ["dvc", "version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[OK] DVC is installed: {result.stdout.split()[0]}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] DVC is not installed")
        print("        Install with: pip install dvc")
        return False

def init_dvc():
    """Initialize DVC in the repository"""
    if os.path.exists(".dvc"):
        print("[INFO] DVC already initialized")
        return True
    
    print("[*] Initializing DVC...")
    result = run_command("dvc init")
    if result is not None:
        print("[OK] DVC initialized successfully")
        return True
    return False

def setup_local_remote():
    """Setup local remote storage for DVC"""
    
    # Create a local remote directory outside the repo
    remote_dir = os.path.abspath("../dvc-storage")
    
    if not os.path.exists(remote_dir):
        os.makedirs(remote_dir)
        print(f"[OK] Created local remote directory: {remote_dir}")
    else:
        print(f"[INFO] Local remote directory exists: {remote_dir}")
    
    # Configure DVC remote
    print("[*] Configuring DVC remote...")
    run_command(f'dvc remote add -d myremote "{remote_dir}"')
    run_command('dvc remote modify myremote cache false')
    
    print("[OK] DVC remote configured")
    return remote_dir

def add_dataset_to_dvc():
    """Add dataset to DVC tracking"""
    
    dataset_file = "dataset/creditcard.csv"
    
    if not os.path.exists(dataset_file):
        print(f"[ERROR] Dataset not found: {dataset_file}")
        print("        Run download_dataset_gdrive.py first to create the dataset")
        return False
    
    # Check if already tracked
    if os.path.exists(f"{dataset_file}.dvc"):
        print(f"[INFO] Dataset already tracked by DVC: {dataset_file}.dvc")
        return True
    
    print(f"[*] Adding dataset to DVC: {dataset_file}")
    result = run_command(f'dvc add "{dataset_file}"')
    
    if result is not None:
        size_mb = os.path.getsize(dataset_file) / (1024 * 1024)
        print(f"[OK] Dataset added to DVC ({size_mb:.2f} MB)")
        print(f"     Created: {dataset_file}.dvc")
        return True
    
    return False

def update_gitignore():
    """Update .gitignore to exclude dataset but track .dvc files"""
    
    gitignore_path = ".gitignore"
    
    # Read current gitignore
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            content = f.read()
    else:
        content = ""
    
    # Check if dataset is already ignored
    if "/dataset/creditcard.csv" not in content:
        print("[*] Updating .gitignore...")
        
        # Remove .dvc from gitignore if present
        lines = content.split('\n')
        lines = [line for line in lines if line.strip() != '.dvc']
        
        # Add dataset to gitignore
        if "# DVC tracked files" not in content:
            lines.append("\n# DVC tracked files")
            lines.append("/dataset/creditcard.csv")
        
        with open(gitignore_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print("[OK] Updated .gitignore")
    else:
        print("[INFO] .gitignore already configured")

def push_to_remote():
    """Push dataset to DVC remote"""
    print("[*] Pushing dataset to DVC remote...")
    result = run_command("dvc push")
    
    if result is not None:
        print("[OK] Dataset pushed to remote storage")
        return True
    return False

def create_download_script():
    """Create a simple script to download dataset using DVC"""
    
    script_content = '''#!/usr/bin/env python3
"""
Download dataset using DVC
This script pulls the dataset from DVC remote storage
"""

import subprocess
import sys
import os

def main():
    print("=" * 60)
    print("DVC Dataset Downloader")
    print("=" * 60)
    
    # Check if DVC is installed
    try:
        subprocess.run(["dvc", "version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] DVC is not installed")
        print("        Install with: pip install dvc")
        sys.exit(1)
    
    # Pull dataset
    print("[*] Pulling dataset from DVC remote...")
    try:
        result = subprocess.run(
            ["dvc", "pull", "dataset/creditcard.csv.dvc"],
            capture_output=True,
            text=True,
            check=True
        )
        print("[OK] Dataset downloaded successfully")
        
        # Verify
        if os.path.exists("dataset/creditcard.csv"):
            size_mb = os.path.getsize("dataset/creditcard.csv") / (1024 * 1024)
            print(f"[OK] Dataset ready: {size_mb:.2f} MB")
            return True
        else:
            print("[ERROR] Dataset file not found after pull")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] DVC pull failed: {e.stderr}")
        print("[INFO] Trying fallback: merge local chunks...")
        
        # Fallback to local chunks
        try:
            import glob
            chunks = sorted(glob.glob("dataset_chunks/creditcard_part_*.csv"))
            if len(chunks) == 8:
                print(f"[*] Found {len(chunks)} chunks, merging...")
                
                with open("dataset/creditcard.csv", 'w', encoding='utf-8') as outfile:
                    first = True
                    for chunk in chunks:
                        with open(chunk, 'r', encoding='utf-8') as infile:
                            if first:
                                outfile.write(infile.read())
                                first = False
                            else:
                                next(infile)
                                outfile.write(infile.read())
                
                print("[OK] Dataset merged from local chunks")
                return True
            else:
                print(f"[ERROR] Not enough chunks found ({len(chunks)}/8)")
                return False
        except Exception as ex:
            print(f"[ERROR] Fallback failed: {ex}")
            return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
    
    with open("download_dataset_dvc.py", 'w') as f:
        f.write(script_content)
    
    print("[OK] Created download_dataset_dvc.py")

def main():
    """Main setup function"""
    
    print("=" * 60)
    print("DVC Setup for Fraud Detection MLOps POC")
    print("=" * 60)
    print()
    
    # Step 1: Check DVC installation
    if not check_dvc_installed():
        print("\n[*] Installing DVC...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "dvc"])
        print("[OK] DVC installed")
    
    # Step 2: Initialize DVC
    if not init_dvc():
        print("[ERROR] Failed to initialize DVC")
        return False
    
    # Step 3: Setup local remote
    remote_dir = setup_local_remote()
    
    # Step 4: Add dataset to DVC
    if not add_dataset_to_dvc():
        print("[ERROR] Failed to add dataset to DVC")
        return False
    
    # Step 5: Update gitignore
    update_gitignore()
    
    # Step 6: Push to remote
    if not push_to_remote():
        print("[WARNING] Failed to push to remote (this is OK for first setup)")
    
    # Step 7: Create download script
    create_download_script()
    
    print("\n" + "=" * 60)
    print("DVC Setup Complete!")
    print("=" * 60)
    print(f"\nLocal remote: {remote_dir}")
    print("\nNext steps:")
    print("1. Commit DVC files to git:")
    print("   git add .dvc .dvcignore dataset/creditcard.csv.dvc .gitignore")
    print("   git commit -m 'Setup DVC for dataset management'")
    print("\n2. To download dataset on another machine:")
    print("   python download_dataset_dvc.py")
    print("\n3. For CI/CD, use: dvc pull dataset/creditcard.csv.dvc")
    print("\nNote: The local remote is at ../dvc-storage")
    print("      For production, configure S3/Azure/GCS remote")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
