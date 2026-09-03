"""
Loads unstructured commentary notes (.docx) and extracts the text + the date
the note is about.

Design note: these notes don't come with structured metadata -- the date is
buried inside the first line of prose ("Satsang 30/08/2026"). Real-world text
data is almost always like this: the structure you need is IN the content,
not attached as a clean field. Extracting it reliably (and failing loudly
when you can't) is most of the work in any "unstructured data" pipeline.
"""
import re
from datetime import date
from pathlib import Path

import docx  # python-docx

NOTES_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "notes" / "raw"

# Matches DD/MM/YYYY or D/M/YYYY -- these notes use Indian date conventions
# (day before month). This assumption is worth stating explicitly: DD/MM and
# MM/DD are silently ambiguous for any date where day <= 12, and guessing
# wrong corrupts every downstream feature without erroring.
DATE_PATTERN = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")


def extract_text(path: Path) -> str:
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs if p.text.strip())


def extract_note_date(text: str) -> date:
    match = DATE_PATTERN.search(text)
    if not match:
        raise ValueError(
            "Could not find a DD/MM/YYYY date in the note text. "
            "Without a reliable date, this note can't be placed on a timeline -- "
            "better to fail here than silently mis-date it."
        )
    day, month, year = (int(g) for g in match.groups())
    return date(year, month, day)


def load_note(path: Path) -> dict:
    text = extract_text(path)
    note_date = extract_note_date(text)
    return {"date": note_date, "source_file": path.name, "text": text}


def load_notes_folder(folder: Path = NOTES_RAW_DIR) -> list[dict]:
    notes = []
    for path in sorted(folder.glob("*.docx")):
        try:
            notes.append(load_note(path))
        except ValueError as e:
            print(f"Skipping {path.name}: {e}")
    notes.sort(key=lambda n: n["date"])
    return notes


if __name__ == "__main__":
    for note in load_notes_folder():
        print(f"{note['date']}  ({note['source_file']})  {len(note['text'])} chars")
