"""Telugu terms to keep as English names before machine translation.

NLLB translates names and Sanskrit terms word by word ("Soundarya Lahari" becomes "the beauty of
Lahri", "Brahma Vidya" becomes "Brahmin education"). Replacing them with their usual English
spelling first makes the model copy them through unchanged.

Users can add their own entries in <user data>/glossary-te.txt, one per line:
    సౌందర్య లహరి = Soundarya Lahari
Lines starting with # are ignored. Entries there are matched as written (plus any Telugu suffix).
"""
from __future__ import annotations

import os
import re

from .tools import user_data_dir

# Any run of Telugu letters/vowel signs: catches inflected endings (-లో, -ని, -కు, ...). Python's
# \w does not match Telugu vowel signs, so the block is spelt out.
TE = "[ఀ-౿]*"
SP = r"\s*"

# (pattern, English). Patterns include common speech-recognition misspellings seen in practice.
BUILTIN_TE = [
    (rf"సౌందర్య{SP}(?:లహరి|లహర|లకరి|లక్షల|లకర|లఖర|లకర్ష){TE}|సౌందర్ల{TE}|సౌందరలక{TE}", "Soundarya Lahari"),
    (rf"శివానంద{SP}లహరి{TE}", "Shivananda Lahari"),
    (rf"(?:ఆది{SP})?శంకర{TE}{SP}భగ{TE}|శంకరాచార్య{TE}|భగవత్పాద{TE}|భగవత్పాల{TE}", "Adi Shankaracharya"),
    (rf"శ్రీ{SP}విద్య{TE}|శ్రీవిద్య{TE}|సిరివిద్య{TE}", "Sri Vidya"),
    (rf"బ్రహ్మ{SP}విద్య{TE}", "Brahma Vidya"),
    (rf"బ్రహ్మ{SP}జ్ఞాన{TE}", "Brahma Jnana"),
    (rf"ఆత్మ{SP}విద్య{TE}", "Atma Vidya"),
    (rf"మహా{SP}విద్య{TE}", "Maha Vidya"),
    (rf"శ్రీ{SP}చక్ర{TE}", "Sri Chakra"),
    (rf"ఉపనిష{TE}|ఉపరిష{TE}", "Upanishad"),
    (rf"ఉపాసన{TE}", "upasana"),
    (rf"సమయాచార{TE}", "Samayachara"),
    (rf"దక్షిణాచార{TE}", "Dakshinachara"),
    (rf"వామాచార{TE}", "Vamachara"),
    (rf"కౌలాచార{TE}", "Kaulachara"),
    (rf"లలితా{SP}(?:దేవి|దేవ|అంబ|ంబ){TE}|లలితాంబ{TE}", "Lalita Devi"),
    (rf"అమ్మవార{TE}", "the Divine Mother"),
    (rf"పరమేశ్వర{TE}", "Parameshvara"),
    (rf"నందీశ్వర{TE}|నందిశ్వర{TE}", "Nandishvara"),
    (rf"కైలాస{TE}", "Kailasa"),
    (rf"కనకధార{TE}{SP}స్తోత్ర{TE}|కనకదార{TE}{SP}స్తోత్ర{TE}", "Kanakadhara Stotram"),
    (rf"అన్నపూర్ణాష్టక{TE}|అన్నపూర్{SP}నష్టక{TE}", "Annapurnashtakam"),
    (rf"భవాని{SP}భుజంగ{TE}", "Bhavani Bhujangam"),
    (rf"బీజాక్షర{TE}|బీదాక్షర{TE}", "bija aksharas"),
    (rf"శ్లోక{TE}", "shloka"),
    (rf"మంత్ర{SP}శాస్త్ర{TE}", "Mantra Shastra"),
    (rf"దీక్ష{TE}", "diksha"),
    (rf"మోక్ష{TE}", "moksha"),
]


def _user_entries(language: str) -> list[tuple[str, str]]:
    path = os.path.join(user_data_dir(), f"glossary-{language}.txt")
    entries = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                source, english = (s.strip() for s in line.split("=", 1))
                if source and english:
                    words = [re.escape(w) for w in source.split()]
                    entries.append((SP.join(words) + TE, english))
    except OSError:
        pass
    return entries


class Glossary:
    def __init__(self, language: str = "te"):
        builtin = BUILTIN_TE if language == "te" else []
        # User entries first, so they win over the built-in ones.
        self.rules = [(re.compile(p), e) for p, e in _user_entries(language) + builtin]

    def protect(self, text: str) -> str:
        for pattern, english in self.rules:
            text = pattern.sub(english, text)
        return text
