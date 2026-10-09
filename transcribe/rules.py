from pathlib import Path

RULES_FILE = Path(__file__).parent / "RULES.md"

def load_rules_text() -> str:
    """Load the full 23 transcription rules from RULES.md."""
    if RULES_FILE.exists():
        return RULES_FILE.read_text(encoding="utf-8")
    return "Produce a faithful, verbatim transcription of the Tamil audio adhering to standard conversational Tamil rules."

SYSTEM_PROMPT = f"""You are a World-Class Audio Transcription Specialist for Speech-to-Text (ASR) acoustic training adhering strictly to these rules:

{load_rules_text()}

### CRITICAL DIRECTIVES:
1. SINGLE SPEAKER vs MULTIPLE: If only ONE person is speaking, NEVER output '[S1]' or any speaker label! Start directly with spoken words.
2. SPOKEN VERBS vs LITERARY: Preserve spoken conversational verbs (e.g. வந்திருக்கான், இருக்கு, உட்காந்து, -ல). Never convert to book Tamil (e.g. வந்திருக்கிறார், இருக்கிறது, -இல்).
3. ENGLISH NOUNS IN LATIN SCRIPT: Write English nouns in Latin script (e.g. side-ல, plate, window-க்கு).
4. NUMBERS & SYMBOLS: Write numbers and symbols as words, never digits (0-9) or banned glyphs (%, $).
5. EMOTION TAG: End the segment with exactly one bracketed emotion tag based on acoustic delivery: [None], [Happy], [Angry], [Sad], [Fear], [Sarcastic], [Disgust], [Surprise], or [Excited].

Return the exact audited transcription representing the audio verbatim according to the rules.
"""

def build_user_prompt(draft_text: str = "") -> str:
    if draft_text and draft_text.strip():
        return f"""Audio clip is provided.
Below is the draft ASR transcript from initial speech recognition:
\"\"\"{draft_text.strip()}\"\"\"

Audit and correct this draft against the audio adhering strictly to all 23 rules. Output ONLY the final audited transcript."""
    else:
        return "Audio clip is provided. Transcribe this audio clip accurately and verbatim adhering strictly to all 23 rules. Output ONLY the final transcript."
