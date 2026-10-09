"""Tennis player identity.

Sources spell names differently:
    ESPN / odds API: "Alex de Minaur", "Carlos Alcaraz", "Zhizhen Zhang"
    tennis-data.co.uk: "De Minaur A.",  "Alcaraz C.",     "Zhang Zh."

Every player is stored under ONE display name, the tennis-data style
"Surname F.", because that's what the history uses. `player_key` reduces any
spelling to surname + first initial ("deminaur_a") so names from different
sources can be matched.
"""
import re
import unicodedata


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def split_name(name: str) -> tuple[str, str]:
    """(surname, first_initial) from either name style."""
    name = re.sub(r"\s+", " ", _ascii(name or "").strip())
    parts = name.split(" ")
    if not name:
        return "", ""
    # tennis-data style: last token is initials like "A." or "Zh." or "J.L."
    if len(parts) > 1 and re.fullmatch(r"([A-Za-z]{1,3}\.)+", parts[-1]):
        return " ".join(parts[:-1]), parts[-1][0]
    if len(parts) == 1:
        return parts[0], ""
    return " ".join(parts[1:]), parts[0][0]


def player_key(name: str) -> str:
    surname, initial = split_name(name)
    return re.sub(r"[^a-z]", "", surname.lower()) + "_" + initial.lower()


def last_word_key(name: str) -> str:
    """Looser key (last word of the surname + initial) for compound surnames
    that one source shortens: 'Martin Etcheverry T.' vs 'Etcheverry T.'"""
    surname, initial = split_name(name)
    last = surname.replace("-", " ").split(" ")[-1] if surname else ""
    return re.sub(r"[^a-z]", "", last.lower()) + "_" + initial.lower()


def display_name(name: str) -> str:
    """'Alex de Minaur' -> 'De Minaur A.'  ('De Minaur A.' is left as is)."""
    surname, initial = split_name(name)
    if not surname:
        return name
    if re.fullmatch(r".*\s([A-Za-z]{1,3}\.)+", _ascii(name.strip())):
        return _ascii(name.strip())
    nice = " ".join(w[:1].upper() + w[1:] for w in surname.split(" "))
    return f"{nice} {initial.upper()}." if initial else nice


class PlayerIndex:
    """Maps any spelling to the display name already used in the database."""

    def __init__(self, known_names):
        self.exact, self.loose = {}, {}
        loose_seen = {}
        for n in known_names:
            if not n:
                continue
            self.exact.setdefault(player_key(n), n)
            k = last_word_key(n)
            loose_seen.setdefault(k, set()).add(n)
        # loose matches only when unambiguous
        self.loose = {k: next(iter(v)) for k, v in loose_seen.items() if len(v) == 1}

    def resolve(self, name: str) -> str:
        return (self.exact.get(player_key(name))
                or self.loose.get(last_word_key(name))
                or display_name(name))


def ordered(p1: str, p2: str) -> tuple[str, str]:
    """Players are always stored alphabetically ('home' = first), so which slot
    a player is in says nothing about who won."""
    return (p1, p2) if p1.lower() <= p2.lower() else (p2, p1)
