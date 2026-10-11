import os
import subprocess
import sys

repo_dir = "/kaggle/working/ASRDATA"
print("=== 1. Git Pull Latest ASRDATA ===")
if os.path.exists(repo_dir):
    res = subprocess.run(["git", "-C", repo_dir, "pull"], capture_output=True, text=True)
    print(res.stdout)
else:
    res = subprocess.run(["git", "clone", "https://github.com/Ragu-123/ASRDATA.git", repo_dir], capture_output=True, text=True)
    print(res.stdout)

print("=== 2. Running segment.py for video ypiPEOLgVDc ===")
cmd = [sys.executable, "-u", f"{repo_dir}/segment.py", "--video-id", "ypiPEOLgVDc", "--shard-threshold", "2"]
res = subprocess.run(cmd, cwd=repo_dir, capture_output=False, text=True)
print(f"=== Completed with exit code: {res.returncode} ===")
