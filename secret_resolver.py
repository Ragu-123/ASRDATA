import os
import sys
from pathlib import Path

# Common aliases for known configuration keys
ALIAS_MAP = {
    "TELEGRAM_BOT_TOKEN": [
        "TELEGRAM_BOT_TOKEN", "telegram_bot_token",
        "BOT_TOKEN", "bot_token",
        "TELEGRAM_TOKEN", "telegram_token",
        "TELEGRAM_API_TOKEN", "telegram_api_token",
        "TG_BOT_TOKEN", "tg_bot_token",
        "TG_TOKEN", "tg_token"
    ],
    "TELEGRAM_ADMIN_ID": [
        "TELEGRAM_ADMIN_ID", "telegram_admin_id",
        "ADMIN_USER_ID", "admin_user_id",
        "TELEGRAM_CHAT_ID", "telegram_chat_id",
        "CHAT_ID", "chat_id",
        "ADMIN_ID", "admin_id",
        "TG_ADMIN_ID", "tg_admin_id",
        "TG_CHAT_ID", "tg_chat_id"
    ],
    "HF_TOKEN": [
        "HF_TOKEN", "hf_token",
        "HUGGINGFACE_TOKEN", "huggingface_token",
        "HF_API_KEY", "hf_api_key",
        "HUGGING_FACE_HUB_TOKEN", "hugging_face_hub_token"
    ],
    "TELEGRAM_ADMIN_PASSCODE": [
        "TELEGRAM_ADMIN_PASSCODE", "telegram_admin_passcode",
        "ADMIN_PASSCODE", "admin_passcode"
    ],
    "PROXY_TOKEN": [
        "PROXY_TOKEN", "proxy_token",
        "GEMINI_PROXY_TOKEN", "gemini_proxy_token"
    ],
}

def resolve_secret(key: str, default: str = "") -> str:
    """
    Robust secret resolver:
    1. Checks os.getenv(key) and known alias variants.
    2. If running on Kaggle (/kaggle/working), checks UserSecretsClient().get_secret() for key & aliases.
    3. Checks local .env files in repository / current working directory.
    4. Returns default if not found.
    """
    keys_to_check = [key]
    if key in ALIAS_MAP:
        for alias in ALIAS_MAP[key]:
            if alias not in keys_to_check:
                keys_to_check.append(alias)
    if key.lower() not in keys_to_check:
        keys_to_check.append(key.lower())
    if key.upper() not in keys_to_check:
        keys_to_check.append(key.upper())

    # 1. Environment variable
    for k in keys_to_check:
        val = os.getenv(k)
        if val:
            cleaned = val.strip()
            if cleaned:
                os.environ[key] = cleaned
                return cleaned

    # 2. Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        for k in keys_to_check:
            try:
                val = user_secrets.get_secret(k)
                if val:
                    val_str = str(val).strip()
                    if val_str:
                        os.environ[key] = val_str
                        return val_str
            except Exception:
                continue
    except Exception:
        pass

    # 3. Local .env file
    candidate_paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
        Path("/kaggle/working/.env"),
        Path("/kaggle/working/NEWasr/.env"),
        Path("/kaggle/working/ASRDATA/.env")
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
                        if k.strip() in keys_to_check:
                            cleaned_val = v.strip().strip("'\"")
                            if cleaned_val:
                                os.environ[key] = cleaned_val
                                return cleaned_val
            except Exception:
                pass

    return default
