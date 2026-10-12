import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

# ==============================================================================
# HARDCODED / USER DEFAULT SECRETS
# You can directly enter or paste your tokens here if not using CLI / Kaggle Secrets
# ==============================================================================
HARDCODED_SECRETS: Dict[str, str] = {
    "HF_TOKEN": "",
    "TELEGRAM_ADMIN_ID": "",
    "TELEGRAM_BOT_TOKEN": "",
    "TELEGRAM_ADMIN_PASSCODE": "",
    "PROXY_TOKEN": ""
}

# Common aliases for known configuration keys
ALIAS_MAP: Dict[str, List[str]] = {
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
        "HUGGING_FACE_HUB_TOKEN", "hugging_face_hub_token",
        "TOKEN", "token"
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

def set_secret(key: str, value: str) -> None:
    """Explicitly set or override a secret in runtime and environment."""
    if not key or not value:
        return
    cleaned = str(value).strip().strip("'\"")
    if cleaned:
        os.environ[key] = cleaned
        HARDCODED_SECRETS[key] = cleaned

def update_secrets(secrets_dict: dict) -> None:
    """Bulk update secrets from a dictionary."""
    for k, v in secrets_dict.items():
        if v:
            set_secret(k, v)

def _parse_secret_from_text(content: str, keys_to_check: List[str]) -> Optional[str]:
    """Parse key value from text supporting .env, json, and tokens list = [...] formats."""
    # 1. Try JSON parsing
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            for k in keys_to_check:
                if k in data and str(data[k]).strip():
                    return str(data[k]).strip().strip("'\"")
    except Exception:
        pass

    # 2. Line by line parsing
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Remove trailing commas or brackets
        line = line.rstrip(",").rstrip("]").rstrip(";")
        
        # Match KEY = VALUE or KEY : VALUE or "KEY": "VALUE"
        if "=" in line:
            parts = line.split("=", 1)
        elif ":" in line:
            parts = line.split(":", 1)
        else:
            continue
            
        k = parts[0].strip().strip("'\"")
        v = parts[1].strip().strip("'\"")
        
        # Skip headers like "tokens list = ["
        if k.lower() in ("tokens list", "tokens", "token_list"):
            continue
            
        if k in keys_to_check and v:
            return v
            
    return None

def resolve_secret(key: str, default: str = "") -> str:
    """
    Robust multi-source secret resolver:
    1. Checks os.getenv(key) and known alias variants.
    2. Checks local token files (.env, tokens.txt, tokens.json, token.txt) in cwd, repo, and /kaggle/working.
    3. If running on Kaggle, checks UserSecretsClient().get_secret() for key & aliases (failsafe).
    4. Checks HARDCODED_SECRETS dictionary.
    5. Returns default if not found.
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
            cleaned = val.strip().strip("'\"")
            if cleaned:
                os.environ[key] = cleaned
                return cleaned

    # 2. Local token files (.env, tokens.txt, tokens.json, token.txt, HF_TOKEN.txt)
    candidate_paths = [
        Path.cwd() / ".env",
        Path.cwd() / "tokens.txt",
        Path.cwd() / "tokens.json",
        Path.cwd() / "token.txt",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent / "tokens.txt",
        Path(__file__).resolve().parent.parent / ".env",
        Path("/kaggle/working/.env"),
        Path("/kaggle/working/tokens.txt"),
        Path("/kaggle/working/token.txt"),
        Path("/kaggle/working/tokens.json"),
        Path("/kaggle/working/ASRDATA/.env"),
        Path("/kaggle/working/ASRDATA/tokens.txt"),
        Path("/kaggle/working/HF_TOKEN.txt"),
        Path(f"/kaggle/working/{key}.txt"),
    ]
    for env_file in candidate_paths:
        if env_file.exists() and env_file.is_file():
            try:
                # If file is specifically named <KEY>.txt or token.txt (when checking HF_TOKEN)
                if env_file.name in (f"{key}.txt", f"{key.lower()}.txt", "HF_TOKEN.txt"):
                    content = env_file.read_text(encoding="utf-8").strip()
                    if content and not content.startswith("#"):
                        os.environ[key] = content
                        return content
                if env_file.name == "token.txt" and key in ("HF_TOKEN", "token"):
                    content = env_file.read_text(encoding="utf-8").strip()
                    if content and not content.startswith("#"):
                        os.environ[key] = content
                        return content

                file_content = env_file.read_text(encoding="utf-8")
                matched_val = _parse_secret_from_text(file_content, keys_to_check)
                if matched_val:
                    os.environ[key] = matched_val
                    return matched_val
            except Exception:
                pass

    # 3. Kaggle Secrets (safe per-key attempt with error suppression)
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        for k in keys_to_check:
            try:
                val = user_secrets.get_secret(k)
                if val:
                    val_str = str(val).strip().strip("'\"")
                    if val_str:
                        os.environ[key] = val_str
                        return val_str
            except Exception:
                continue
    except Exception:
        pass

    # 4. HARDCODED_SECRETS Fallback
    for k in keys_to_check:
        if k in HARDCODED_SECRETS and HARDCODED_SECRETS[k]:
            val_str = str(HARDCODED_SECRETS[k]).strip()
            if val_str:
                return val_str

    # 5. Default argument
    if default:
        return default

    return ""

def add_secret_cli_args(parser) -> None:
    """Add standard token/secret arguments to an argparse.ArgumentParser."""
    try:
        group = parser.add_argument_group("Tokens & Secrets")
        group.add_argument(
            "--hf-token", "--token", "--hf_token",
            dest="hf_token",
            type=str,
            default="",
            help="Hugging Face API token (overrides env, Kaggle secrets, and .env)"
        )
        group.add_argument(
            "--telegram-bot-token", "--bot-token", "--telegram_bot_token",
            dest="telegram_bot_token",
            type=str,
            default="",
            help="Telegram Bot token for notifications and cluster control"
        )
        group.add_argument(
            "--telegram-admin-id", "--admin-id", "--telegram_admin_id",
            dest="telegram_admin_id",
            type=str,
            default="",
            help="Telegram Admin chat ID (e.g. 5959376217)"
        )
    except Exception:
        pass

def apply_cli_secrets(args) -> None:
    """Apply CLI-provided secret flags to os.environ and HARDCODED_SECRETS."""
    if getattr(args, "hf_token", None):
        set_secret("HF_TOKEN", args.hf_token)
    if getattr(args, "telegram_bot_token", None):
        set_secret("TELEGRAM_BOT_TOKEN", args.telegram_bot_token)
    if getattr(args, "telegram_admin_id", None):
        set_secret("TELEGRAM_ADMIN_ID", str(args.telegram_admin_id))
