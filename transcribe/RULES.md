# LLM Audio Transcription Instructions

## Purpose

Produce a faithful, corrected transcription of the audio. The transcript must represent **what was actually spoken and heard**, while applying the formatting, script, punctuation, speaker, acoustic-tag, emotion, disfluency, and truncation rules below.

Do **not** paraphrase, summarize, normalize the meaning, or silently improve the speaker's wording. When a decision depends on the audio, the audio is authoritative.

---

# WORD ACCURACY

## 01. Replace Every Wrong Word

If a word in the transcript does not match what is spoken in the audio, replace it with the **exact word spoken**.

Do not replace it with a synonym, paraphrase, or a word that merely makes the sentence sound more natural. Use the precise spoken word.

### Example

**In the transcript:**

> தண்ணீர் வாங்குங்கள்

**You write:**

> தண்ணீர் குடிங்க

### Rule

- Compare every transcribed word against the audio.
- Correct a word only to the word actually heard.
- Never substitute a semantically similar word merely because it fits the context better.

---

## 02. Add Every Missing Word

If a word is clearly spoken but is absent from the transcript, insert it in the **correct position**.

Nothing the speaker clearly said should be missing from the corrected transcript.

### Example

**In the transcript:**

> அவன் கடைக்கு சென்றான்

**You write:**

> அவன் நேற்று கடைக்கு சென்றான்

### Rule

- Listen for omitted words.
- Insert each clearly spoken word where it occurs in the audio.
- Do not add words merely because they are grammatically expected or make the sentence sound better.

---

## 03. Remove Words Not Spoken

If a word appears in the transcript but is **not heard in the audio**, remove it.

Do not keep a word just because it sounds plausible or makes the sentence flow better. Only words actually spoken belong in the transcript.

### Example

**In the transcript:**

> நாளை கண்டிப்பாக வா

**You write:**

> நாளை வா

### Rule

- The audio is the source of truth.
- Remove invented, hallucinated, or transcript-only words.
- Do not preserve a plausible word when it is not actually spoken.

---

## 04. Phonetic Variants

Always write the **standard accepted spelling** of standard vocabulary roots.

Do not invent informal slang misspellings for standard vocabulary words. However, **strictly preserve verbatim spoken conversational Tamil verb and suffix forms** (e.g. `வந்திருக்கான்`, `இருக்கு`, `உட்காந்து`, `குடுத்துருக்காங்க`, `-ல`). Never convert spoken conversational Tamil into formal literary book Tamil (`வந்திருக்கிறார்`, `இருக்கிறது`, `உட்கார்ந்து`, `கொடுத்திருக்கிறார்கள்`, `-இல்`).

### Example

**Wrong (casual slang misspelling of root vocabulary):**

> நெறைய பேர் வந்தாங்க

**Correct (standard lexical spelling with spoken verb):**

> நிறைய பேர் வந்தாங்க

### Rule

- Standard spelling applies to vocabulary roots (write standard `நிறைய` not slang `நெறைய`).
- Strictly preserve the exact spoken verb and grammar: if speaker says `வந்திருக்கான்`, write `வந்திருக்கான்` (NEVER change to `வந்திருக்கிறார்`).
- If speaker says `இருக்கு` / `இருக்குது`, write `இருக்கு` / `இருக்குது` (NEVER change to `இருக்கிறது`).
- If speaker says `உட்காந்து`, write `உட்காந்து` (NEVER change to `உட்கார்ந்து`).
- If speaker says `திறந்துருக்குது`, write `திறந்துருக்குது`.
- If speaker says locative `-la`, write `-ல` (NEVER change to `-இல்` or `-ல்`).

---

# SCRIPT & LANGUAGE SWITCHING

## 05. English Common Nouns & Suffixes

Standard English common nouns such as `side`, `picture`, `window`, `dam`, `plate`, `mobile`, `fan`, `room`, `running`, `square feet`, `laptop`, `table`, `chair`, and similar words must always be written in **Latin/English script**, even when they occur inside a native-language sentence.

Tamil grammatical suffixes attach with a hyphen:
- When the speaker says spoken locative '-la' (ல), attach `-ல` (e.g. `side-ல`, `picture-ல`, `window-ல`, `dam-ல`, `fan-ல`, `room-ல`, `running-ல`, `square feet-ல`).
  * ❌ NEVER write `side-இல்` or `side-ல்` when `side-la` was spoken!
  * ❌ NEVER write `picture-இல்` when `picture-la` was spoken!
- When the speaker says spoken dative '-kku', attach `-க்கு` (e.g. `side-க்கு`, `window-க்கு`, `meeting-க்கு`).

### Example

**Wrong:**

> என்னிடம் பிளேட் இல்லை, அவன் side-இல் நின்றான்

**Correct:**

> என்னிடம் plate இல்லை, அவன் side-ல நின்றான்

### Rule

- Recognize standard English common nouns spoken inside a Tamil sentence and write them in Latin script.
- Do not transliterate them into Tamil script (no `பிளேட்`, `விண்டோ`, `பிக்சர்`, `ஃபேன்`).
- Attach Tamil suffixes with a hyphen, using `-ல` for spoken `-la`.

---

## 06. Proper Nouns in Latin Script

Use the script of whatever name the speaker actually said:
- If spoken in English: write in Latin script (`Chennai`, `Apple`, `Rahul`, `Singapore`, `India`).
- If spoken in Tamil: write in Tamil script (`சென்னையில`, `பாரத்`, `சிங்கப்பூர்`).

### Example

**Speaker says "Chennaila" (Tamil):**
> அவன் சென்னையில போனான்

**Speaker says "Chennai" (English):**
> அவன் Chennai போனான்

### Rule

- If the speaker says the English name, write it in Latin script.
- If the speaker says the Tamil name or uses Tamil suffix (`சென்னையில`), write it in Tamil script.

---

# DIGIT & BANNED SYMBOL RULES

## 07. Numbers as Words, Not Digits

All numbers must be written out as **words**.

Never use numerical digits (`0`–`9`) in the transcript.

### Example

**Wrong:**

> 9 ரூபாய்

**Correct:**

> ஒன்பது ரூபாய்

### Rule

- Convert spoken numbers to words.
- Never leave a digit in the final transcript when a number is spoken.

---

## 08. No Banned Symbols

Special symbols such as `%`, `$`, `#`, `&`, `;`, and `:` are strictly prohibited in the transcript when they represent spoken content.

Write the **spoken equivalent as words** instead.

### Example

**Wrong:**

> ஐம்பது % தள்ளுபடி

**Correct:**

> ஐம்பது சதவீதம் தள்ளுபடி

### Rule

- Never output the prohibited symbol glyph.
- Convert a spoken symbol into the corresponding spoken/written word.
- This applies even when the symbol is commonly used in normal writing.

---

# ACOUSTIC TAGGING RULES

## 09. Hard Acoustic Tags (Music / Overlap)

Distinct, unmistakable background sounds must always be wrapped in their required tags.

Use:

- `<music>...</music>` for music or singing.
- `<overlap>...</overlap>` for overlapping speech.

These tags must be included whenever the acoustic event is clearly present. **Never omit a clearly identifiable hard acoustic event.**

### Example: Music

**Wrong:**

> இன்று வானிலை நன்றாக இருக்கிறது

**Correct:**

> <music> இன்று வானிலை நன்றாக இருக்கிறது </music>

### Rule

- Tag clearly identifiable music, singing, or overlapping speech.
- Put the opening and closing tags around the relevant span.
- Do not silently remove a hard acoustic event.

---

## 10. Soft Acoustic Tags (Noise / No-Speech)

Faint or highly debatable background sounds such as noise or silence are **subjective**.

Possible tags include `<noise>` and `[NO_SPEECH]`.

Whether to include a soft tag is a **judgment call**. Do not force a soft tag when the acoustic event is uncertain or debatable.

### Example

**Wrong:**

> நாளை சந்திப்போம்

**Correct:**

> <noise> நாளை சந்திப்போம் </noise>

### Rule

- Use soft acoustic tags only when the background event is sufficiently meaningful to warrant tagging.
- Distinguish soft, subjective events from hard, unmistakable acoustic events.
- Do not turn every tiny background sound into a tag.

---

# PUNCTUATION RULES

## 11. Question Mark Only for Rising Intonation

A question mark (`?`) may be used **only when the speaker's voice clearly rises in intonation**.

A grammatically phrased question spoken with flat intonation must **not** receive a question mark.

### Example

**Wrong:**

> நீங்கள் போகிறீர்களா?

**Correct:**

> நீங்கள் போகிறீர்களா

### Rule

- Judge punctuation from the **audio intonation**, not grammar alone.
- Rising intonation → `?` is allowed.
- Flat intonation → do not use `?`.

---

## 12. Structural Punctuation (Soft)

Minor differences in everyday punctuation are acceptable stylistic variations.

A sentence may end with a comma, a full stop, or a native ending mark depending on the transcription style. These small structural differences are not the main accuracy target.

### Example

Both forms are acceptable:

> அவன் கடைக்கு போனான்.

> அவன் கடைக்கு போனான்,

### Rule

- Do not over-correct ordinary punctuation differences.
- Prioritize word accuracy and acoustic fidelity.
- The question-mark rule in **Rule 11** is stricter than this soft punctuation rule.

---

# DIARIZATION RULES

## 13. Accurate Speaker Tracking

- **CRITICAL: SINGLE-SPEAKER CLIPS (Applies to >95% of clips):**
  * When only ONE person is speaking in the audio clip, **DO NOT ADD ANY SPEAKER TAG AT ALL**.
  * ❌ NEVER output `[S1]`, `Speaker_1:`, or any speaker label when only one speaker is heard!
  * Start the transcript directly with the words spoken.
  * Example:
    > அப்புறம் side-ல window எல்லாம் நல்லாவே திறந்துருக்குது [None]
    *(Note: ABSOLUTELY NO [S1]!)*

- **MULTIPLE-SPEAKER CLIPS ONLY:**
  * ONLY when two or more distinct individuals speak in the audio clip, label each distinct voice with `[S1]`, `[S2]`, etc.
  * Example (2 distinct speakers conversing):
    > [S1] பெயர் என்ன? [S2] ராஜேஷ். [None]

### Rule

- For single-speaker audio, NO speaker tags (`[S1]`) allowed.
- For multi-speaker audio, assign a separate speaker ID (`[S1]`, `[S2]`) to each distinct voice.
- Keep the same speaker ID consistent across that speaker's turns.
- Never merge two speakers into one label.

---

# EMOTION TAGGING

## 14. Tag the Dominant Emotion

Every speaker segment ends with an emotion tag.

Use one of:

- `[Happy]`
- `[Angry]`
- `[Sad]`
- `[Fear]`
- `[Sarcastic]`
- `[Disgust]`
- `[Surprise]`
- `[Excited]`
- `[None]`

Use one of the named emotions **only when that emotion is audibly dominant in more than 50% of the segment**. Otherwise use `[None]`.

The emotion must be based on the **acoustic delivery you hear**, not on the emotional meaning implied by the words.

### Example

**Wrong:**

> அவன் ஒருபோதும் மன்னிக்க மாட்டேன் என்று கத்தினான் [None]

**Correct:**

> அவன் ஒருபோதும் மன்னிக்க மாட்டேன் என்று கத்தினான் [Angry]

### Rule

- Tag the emotion that dominates the actual voice performance.
- Do not infer an emotion solely from the sentence content.
- Exactly one emotion tag belongs to each speaker segment.
- Use `[None]` when no listed emotion is audibly dominant for more than half of the segment.

---

# ACOUSTIC TAGGING & DISFLUENCY RULES

## 15. Non-lexical Fillers Become Tags

Hesitation sounds must be written as **bracketed tags exactly where they occur**.

Examples:

- `[um]`
- `[uh]`
- `[hmm]`
- `[ah]`

Do not spell these out as ordinary words.

### Example

**Wrong:**

> அதனால் மம் நாம் போகணும்னு நினைக்கிறேன்

**Correct:**

> அதனால் [um] நாம் போகணும்னு நினைக்கிறேன்

- Listen closely for hesitation and thinking sounds: write them as `[um]`, `[uh]`, `[hmm]`, or `[ah]` at the exact point they occur in the audio.
- Never omit non-lexical fillers.
- Conversational discourse words (e.g. `வந்து`, `அப்புறம்`, `அதாவது`) are real Tamil words, NOT bracketed tags—transcribe them verbatim in Tamil script as spoken (e.g. `இந்த picture-ல வந்து தண்ணி கொஞ்சம் தேங்கி நிக்குது`).

---

# VERBATIM & FORMATTING

## 16. Echo / Reduplication Words

Reduplicated words are joined with the specified **tilde separator `~`**, with each half kept in the script in which it was spoken.

### Example

**Wrong:**

> சாப்பாடு - வாப்பாடு சாப்பிட்டாய?

**Correct:**

> சாப்பாடு ~ வாப்பாடு சாப்பிட்டாய?

### Rule

- Preserve both repeated halves.
- Keep each half in the script actually spoken.
- Use `~` between the repeated halves, not a normal hyphen `-`.
- Do not collapse the repeated form into one word or silently remove it.

---

## 17. Spelled-out Letters

Letters spoken one at a time are written **capitalised** and separated by hyphens/spaces as individual letters rather than run together as a normal word.

The intended format is an explicitly separated sequence such as `U-P-S-C`.

### Example

**Wrong:**

> நான் bsnl க்கு போன் பண்ணேன்

**Correct:**

> நான் B S N L க்கு போன் பண்ணேன்

### Rule

- Preserve each spoken letter individually.
- Capitalize letter names when they are spoken as letters.
- In Tamil/native-script utterances, separate the letters with spaces rather than hyphens (`B S N L`).
- Do not merge them into a lowercase word or ordinary lexical spelling.

---

## 18. Time Spelled in Words

Times must be spelled out in **words**, never as digits.

AM/PM are treated like spelled-out letters and should remain separated as individual capital letters.

### Example

**Wrong:**

> நான் 5 PM க்கு வருவேன்

**Correct:**

> நான் அஞ்சு P-M க்கு வருவேன்

### Rule

- Convert numeric time values to words.
- Do not output digit-based times such as `5 PM`, `5:00 PM`, or similar forms.
- Represent spoken `AM` / `PM` as separated letters such as `A-M` / `P-M` according to the formatting convention.

---

## 19. Continuous IDs & Alphanumeric Strings

For an ID or code, each element is spelled out according to the way it is spoken.

- Letters are represented as letters.
- Numbers are represented as words.
- Keep the elements separated rather than reproducing the compact alphanumeric glyph string.

### Example

**Wrong:**

> என் code 6E124

**Correct:**

> என் code six E one two four

### Rule

- Do not copy an ID/code as a raw mixed alphanumeric string when the audio spells its components.
- Preserve letter identity and number identity separately.
- Numbers inside the code must also follow the number-as-words rule.

---

## 20. Spoken Symbols Become Words

Symbols must be written **exactly as spoken**, not as their symbol glyphs.

Examples of the spoken conversions shown in the reference:

- `@` → `at the rate`
- `.` → `dot`
- `www` → `w w w`

Never output the symbol glyph itself when the speaker speaks its name.

### Example

**Wrong:**

> மெயில் பண்ணு test@gmail.com க்கு

**Correct:**

> மெயில் பண்ணு test at the rate gmail dot com க்கு

### Rule

- Transcribe the spoken name of a symbol as words.
- Do not preserve the visual symbol glyph merely because the content is an email address or similar structured text.
- Apply the same principle to other symbols when their spoken form is audible.

---

# DISFLUENCIES & REPAIRS

## 21. Self-correction — Keep Both Attempts

When a speaker corrects themselves, transcribe the **false start AND the correction verbatim**.

Include words such as `sorry` when they are spoken. Do not clean up the speaker's repair.

### Example

**Wrong:**

> என் number ஒன்று எட்டு ஒன்று இரண்டு

**Correct:**

> என் number ஒன்று எட்டு ஒன்று ஒன்று sorry ஒன்று எட்டு ஒன்று இரண்டு

### Rule

- Keep the initial mistaken attempt.
- Keep the corrected attempt.
- Preserve spoken repair language such as `sorry`.
- Never rewrite the sentence as though the speaker had said only the final intended version.

---

## 22. Stutters & False Starts

Mark a repeated initial sound with `=` at the onset.

Do not clean up the stutter.

### Example

**Wrong:**

> நான் செல்லலை

**Correct:**

> ந = நான் செல்லலை

### Rule

- Preserve genuine stuttering and false starts.
- Use `=` to indicate the repeated onset according to the format shown.
- Do not convert a stutter into a fluent final sentence.
- Do not erase the evidence of the repeated initial sound.

---

## 23. Truncations — Never Hallucinate

If a word is cut off before completion, **do not guess the full word**.

Mark the truncation with a **double hyphen** `--`.

### Example

**Wrong:**

> இன்று ரொம்ப important

**Correct:**

> இன்று ரொம்ப impor--

### Rule

- Write only the portion actually heard.
- End the incomplete word with `--`.
- Never infer the remainder from context.
- Never complete a truncated word just because the intended word seems obvious.

---

# FINAL TRANSCRIPTION CHECKLIST

Before returning the final transcript, verify all **23 rules** against the audio:

1. Wrong words were replaced with the exact spoken words.
2. Missing spoken words were inserted.
3. Unspoken words were removed.
4. Standard spelling was used instead of phonetic spellings.
5. English common nouns stayed in Latin script.
6. Designated English proper nouns stayed in Latin script.
7. Numbers were written as words, not digits.
8. Banned symbol glyphs were not used for spoken equivalents.
9. Hard acoustic events were tagged when unmistakable.
10. Soft acoustic tags were handled as a judgment call.
11. `?` was used only for clearly rising intonation.
12. Ordinary punctuation differences were treated as soft/stylistic.
13. Every distinct speaker received a separate speaker label.
14. Every speaker segment ended with the correct dominant-emotion tag.
15. Non-lexical fillers were represented with bracketed filler tags.
16. Echo/reduplication formatting was preserved.
17. Spelled-out letters were kept separated and capitalized.
18. Times were written in words, with AM/PM treated as letters.
19. IDs and alphanumeric strings were expanded according to how they were spoken.
20. Spoken symbol names were written as words rather than glyphs.
21. Self-corrections kept both the false start and the correction.
22. Stutters and false starts were preserved with the required onset marker.
23. Truncated words were marked with `--` and never completed by guesswork.

## Highest-priority principle

**The audio is authoritative.**

When the existing transcript conflicts with what is actually heard, correct the transcript to the audio while preserving the required script, formatting, speaker, acoustic, emotion, disfluency, and punctuation conventions above.

Do not hallucinate missing speech. Do not paraphrase. Do not silently clean up speech that the speaker actually produced.
