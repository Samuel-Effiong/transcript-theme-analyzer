"""Theme list input: one text per theme (a phrase or a whole paragraph), a
derived short label, unique ids and the MAX_THEMES limit."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass

LABEL_WORDS = 6
_TRAILING_PUNCTUATION = ".,;:!?\u2014\u2013-\"'"


class ThemeInputError(ValueError):
    """Bad theme input. The message is the user-facing copy."""


@dataclass(frozen=True)
class Theme:
    """``text`` is the theme as the user wrote it and as the model reads it.
    ``name`` is the short label used for file names, headings and headers;
    it defaults to the text itself."""
    id: str
    name: str
    text: str = ""

    def __post_init__(self) -> None:
        if not self.text:
            object.__setattr__(self, "text", self.name)


def slugify(name: str) -> str:
    # Apostrophes vanish rather than hyphenate: "God's will" -> "gods-will".
    slug = re.sub(r"[^a-z0-9]+", "-", re.sub(r"['\u2019]", "", name.lower())).strip("-")
    return slug or "theme"


def _one_line(text: str) -> str:
    """Runs of whitespace (newlines included) -> one space: each theme must
    stay on one line of the prompt's THEMES block."""
    return " ".join(text.split())


def derive_label(text: str, max_words: int = LABEL_WORDS) -> str:
    """The first ``max_words`` words of ``text``, trailing punctuation removed."""
    words = text.split()[:max_words]
    return " ".join(words).rstrip(_TRAILING_PUNCTUATION)


def build_themes(cli_texts: list[str], file_texts: list[str]) -> list[Theme]:
    """Command-line texts first, then file texts. Labels and ids are unique."""
    entries = list(cli_texts) + list(file_texts)
    if not entries:
        raise ThemeInputError("No themes given. Use --theme or --themes-file.")

    themes: list[Theme] = []
    seen_texts: set[str] = set()
    used_labels: set[str] = set()
    used_ids: set[str] = set()
    for raw in entries:
        text = _one_line(raw)
        if not text:
            raise ThemeInputError("A theme is empty.")
        key = text.casefold()
        if key in seen_texts:
            raise ThemeInputError(f"Theme {text!r} is listed more than once.")
        seen_texts.add(key)

        base_label = derive_label(text)
        label, n = base_label, 1
        while label.casefold() in used_labels:
            n += 1
            label = f"{base_label} ({n})"
        used_labels.add(label.casefold())

        base_id = slugify(label)
        theme_id, n = base_id, 1
        while theme_id in used_ids:
            n += 1
            theme_id = f"{base_id}-{n}"
        used_ids.add(theme_id)
        themes.append(Theme(id=theme_id, name=label, text=text))
    return themes


def read_themes_file(path: str) -> list[str]:
    """Read a CSV with one ``theme`` column; each non-blank cell is a theme."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = {(h or "").strip().lower(): h for h in (reader.fieldnames or [])}
        if "description" in fields:
            raise ThemeInputError(
                f"{path} has a 'description' column. Put the full theme text in the "
                "'theme' column instead; a theme may be a phrase or a whole paragraph."
            )
        if "theme" not in fields:
            raise ThemeInputError(f"{path} needs a 'theme' column.")
        theme_col = fields["theme"]
        texts: list[str] = []
        for row in reader:
            text = (row.get(theme_col) or "").strip()
            if text:
                texts.append(text)
    return texts


def check_theme_limit(themes: list[Theme], max_themes: int) -> None:
    n = len(themes)
    if n > max_themes:
        raise ThemeInputError(
            f"You entered {n} themes. The limit is {max_themes} (MAX_THEMES). "
            f"Remove {n - max_themes} theme(s) or raise MAX_THEMES."
        )
