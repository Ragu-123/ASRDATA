import os
import sys
from pathlib import Path

def resolve_secret(key: str, default: str = "") -> str:
    """
    Robust secret resolver:
    1. Checks os.getenv(key).
    2. If running on Kaggle (/kaggle/working), checks UserSecretsClient().get_secret(key).
    3. Checks local .env files in repository / current working directory.
    4. Returns default if not found.
    """
    # 1. Environment variable
    val = os.getenv(key)
    if val:
        return val.strip()

    # 2. Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        val = user_secrets.get_secret(key)
        if val:
            val_str = str(val).strip()
            os.environ[key] = val_str
            return val_str
    except Exception:
        pass

    # 3. Local .env file
    candidate_paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env"
    ]
    for env_file in candidate_paths:
        if env_file.exists() and env_file.is_file():
            try:
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == key:
                            cleaned_val = v.strip().strip("'\"")
                            os.environ[key] = cleaned_val
                            return cleaned_val
            except Exception:
                pass

    return default
