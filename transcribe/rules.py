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
5. MUSIC & ACOUSTIC RULES:
   - If there are NO words and ONLY music or song is playing: output exactly '[music]'. Never hallucinate words or output empty text.
   - If music is playing in the background while speech is occurring: wrap the spoken speech with <music>...spoken text...</music> (e.g. <music>இன்று வானிலை நன்றாக இருக்கிறது</music> [None]).
6. EMOTION TAG: For speech clips, end the segment with exactly one bracketed emotion tag based on acoustic delivery: [None], [Happy], [Angry], [Sad], [Fear], [Sarcastic], [Disgust], [Surprise], or [Excited]. If the clip is pure music with NO speech, output ONLY '[music]' without an emotion tag.

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

def build_bundle_prompt(timestamps: list, clips: list) -> str:
    """Build prompt for bundled multi-clip transcription with [#cut] delimiter."""
    N = len(timestamps)
    timing_lines = []
    for idx, (st, en) in enumerate(timestamps, start=1):
        draft = clips[idx - 1].get("draft_text", "")
        hint = f" (Draft: \"{draft}\")" if draft and draft.strip() else ""
        timing_lines.append(f"Segment {idx}: [{st:.2f}s - {en:.2f}s]{hint}")
    timings_str = "\n".join(timing_lines)

    return f"""The attached audio contains {N} speech segments concatenated together.
Segment timestamps:
{timings_str}

CRITICAL INSTRUCTIONS:
1. Transcribe each segment verbatim according to the 23 Tamil ASR rules.
2. Output the transcription for each segment in exact sequential order.
3. Separate each segment's transcription using the delimiter '[#cut]' on its own line:
<Segment 1 Transcript>
[#cut]
<Segment 2 Transcript>
[#cut]
...
<Segment {N} Transcript>
4. You MUST output exactly {N} segment transcripts separated by [#cut].
5. If a segment has ONLY music and no speech, output '[music]'.
6. For speech segments, end with the appropriate emotion tag: [None], [Happy], [Angry], [Sad], [Fear], [Sarcastic], [Disgust], [Surprise], or [Excited].
7. Output ONLY the transcripts separated by '[#cut]'. Do NOT include segment numbers, markdown formatting, or notes."""
