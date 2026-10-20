"""Theme list input: parsing, id generation and the MAX_THEMES limit."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass


class ThemeInputError(ValueError):
    """Bad theme input. The message is the user-facing copy."""


@dataclass(frozen=True)
class Theme:
    id: str
    name: str
    description: str = ""


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "theme"


def _one_line(text: str) -> str:
    """Runs of whitespace (newlines included) -> one space: each theme must
    stay on one line of the prompt's THEMES block."""
    return " ".join(text.split())


def build_themes(names: list[str], file_rows: list[tuple[str, str]]) -> list[Theme]:
    """Command-line names first, then file rows. Ids are unique slugs."""
    entries = [(n, "") for n in names] + list(file_rows)
    if not entries:
        raise ThemeInputError("No themes given. Use --theme or --themes-file.")

    themes: list[Theme] = []
    seen_names: set[str] = set()
    used_ids: set[str] = set()
    for raw_name, description in entries:
        name = _one_line(raw_name)
        if not name:
            raise ThemeInputError("A theme name is empty.")
        key = name.casefold()
        if key in seen_names:
            raise ThemeInputError(f"Theme {name!r} is listed more than once.")
        seen_names.add(key)

        base = slugify(name)
        theme_id, n = base, 1
        while theme_id in used_ids:
            n += 1
            theme_id = f"{base}-{n}"
        used_ids.add(theme_id)
        themes.append(Theme(id=theme_id, name=name, description=_one_line(description)))
    return themes


def read_themes_file(path: str) -> list[tuple[str, str]]:
    """Read a CSV with a ``theme`` column and an optional ``description``."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = {(h or "").strip().lower(): h for h in (reader.fieldnames or [])}
        if "theme" not in fields:
            raise ThemeInputError(
                f"{path} needs a 'theme' column (and optionally 'description')."
            )
        theme_col = fields["theme"]
        desc_col = fields.get("description")
        rows: list[tuple[str, str]] = []
        for row in reader:
            name = (row.get(theme_col) or "").strip()
            description = ((row.get(desc_col) or "") if desc_col else "").strip()
            if not name and not description:
                continue
            rows.append((name, description))
    return rows


def check_theme_limit(themes: list[Theme], max_themes: int) -> None:
    n = len(themes)
    if n > max_themes:
        raise ThemeInputError(
            f"You entered {n} themes. The limit is {max_themes} (MAX_THEMES). "
            f"Remove {n - max_themes} theme(s) or raise MAX_THEMES."
        )
