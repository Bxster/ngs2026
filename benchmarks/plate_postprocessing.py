#!/usr/bin/env python3
"""
Post-processing OCR per targhe italiane.

Cascata:
  1. Cleanup char (A-Z + 0-9)
  2. Rimozione 'IT' iniziale (banda blu UE)
  3. Test pattern moto AA12345
  4. Test pattern auto storica XX123456 (con sigla provincia valida)
  5. Default: sliding window con bonus AA000AA + correzione I/O/Q->1/0
     nelle posizioni cifra (2,3,4) prima dello scoring
"""

import re


def clean_ocr_text(text):
    """Tieni solo A-Z e 0-9 maiuscoli."""
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def strip_it_prefix(text):
    """Rimuove 'IT' a inizio stringa (banda blu UE)."""
    if text.startswith("IT") and len(text) > 2:
        return text[2:]
    return text


RE_MOTO = re.compile(r'^[A-Z]{2}[0-9]{5}$')
RE_STORICA = re.compile(r'^[A-Z]{2}[0-9]{6}$')

SIGLE_PROVINCE = {
    "AG", "AL", "AN", "AO", "AP", "AQ", "AR", "AT", "AV",
    "BA", "BG", "BI", "BL", "BN", "BO", "BR", "BS", "BT", "BZ",
    "CA", "CB", "CE", "CH", "CL", "CN", "CO", "CR", "CS", "CT", "CZ",
    "EN",
    "FC", "FE", "FG", "FI", "FM", "FR",
    "GE", "GO", "GR",
    "IM", "IS",
    "KR",
    "LC", "LE", "LI", "LO", "LT", "LU",
    "MB", "MC", "ME", "MI", "MN", "MO", "MS", "MT",
    "NA", "NO", "NU",
    "OR",
    "PA", "PC", "PD", "PE", "PG", "PI", "PN", "PO", "PR", "PT", "PU", "PV", "PZ",
    "RA", "RC", "RE", "RG", "RI", "RM", "RN", "RO",
    "SA", "SI", "SO", "SP", "SR", "SS", "SV",
    "TA", "TE", "TN", "TO", "TP", "TR", "TS", "TV",
    "UD",
    "VA", "VB", "VC", "VE", "VI", "VR", "VS", "VT", "VV",
}

# Mappa di correzione lettera->cifra per le posizioni dove ci aspettiamo cifre
LETTER_TO_DIGIT_FIX = {
    'I': '1',
    'O': '0',
    'Q': '0',
    'L': '4',
    'T': '7',
}


def looks_like_moto(text):
    return bool(RE_MOTO.match(text))


def looks_like_storica(text):
    if not RE_STORICA.match(text):
        return False
    return text[:2] in SIGLE_PROVINCE


def find_pattern_substring(text, length, validator):
    for i in range(len(text) - length + 1):
        sub = text[i:i + length]
        if validator(sub):
            return sub
    return None


def fix_letters_in_digit_positions(s):
    """
    Per stringa di 7 char, sostituisce I/O/Q con 1/0 nelle posizioni cifra (2,3,4)
    del pattern AA000AA. Le altre posizioni restano invariate.
    Ritorna la versione corretta (anche se identica all'originale).
    """
    if len(s) != 7:
        return s
    fixed = list(s)
    for i in [2, 3, 4]:
        if fixed[i] in LETTER_TO_DIGIT_FIX:
            fixed[i] = LETTER_TO_DIGIT_FIX[fixed[i]]
    return ''.join(fixed)


def score_candidate(s):
    """
    Score per sliding window.
    Bonus formato italiano AA000AA, extra se posizioni lettera sono pulite (no I/O/U/Q).
    """
    if not s:
        return -999
    n_letters = sum(c.isalpha() for c in s)
    n_digits = sum(c.isdigit() for c in s)
    length_penalty = -abs(len(s) - 7) * 0.5
    mix_score = min(n_letters, n_digits) * 1.0

    italian_bonus = 0.0
    if len(s) == 7:
        pos_letters = [s[0], s[1], s[5], s[6]]
        pos_digits = [s[2], s[3], s[4]]
        if all(c.isalpha() for c in pos_letters) and all(c.isdigit() for c in pos_digits):
            italian_bonus = 3.0
            if not any(c in "IOUQ" for c in pos_letters):
                italian_bonus += 2.0

    return length_penalty + mix_score + italian_bonus


def sliding_window_best(text, min_len=5, max_len=9):
    """
    Genera tutte le sottostringhe di lunghezza min_len..max_len.
    Per ogni sottostringa di lunghezza 7, genera anche la versione 'corretta'
    con I/O/Q->1/0 nelle posizioni cifra.
    Ritorna il candidato (originale o corretto) con score massimo.
    """
    if not text:
        return ""
    candidates = [text]
    for length in range(min_len, max_len + 1):
        for i in range(len(text) - length + 1):
            sub = text[i:i + length]
            candidates.append(sub)
            # Per i candidati di lunghezza 7, genera anche la variante corretta
            if length == 7:
                fixed = fix_letters_in_digit_positions(sub)
                if fixed != sub:
                    candidates.append(fixed)
    return max(candidates, key=score_candidate)


def is_auto_moderna(s):
    if len(s) != 7:
        return False
    pos_letters = [s[0], s[1], s[5], s[6]]
    pos_digits = [s[2], s[3], s[4]]
    return all(c.isalpha() for c in pos_letters) and all(c.isdigit() for c in pos_digits)


def postprocess_plate(raw_text):
    """
    Ritorna {'text': str, 'format': str}.
    format: 'moto' | 'auto_storica' | 'auto_moderna' | 'unknown'.
    """
    if not raw_text:
        return {'text': '', 'format': 'unknown'}

    cleaned = clean_ocr_text(raw_text)
    if not cleaned:
        return {'text': '', 'format': 'unknown'}

    cleaned = strip_it_prefix(cleaned)

    if looks_like_moto(cleaned):
        return {'text': cleaned, 'format': 'moto'}

    if looks_like_storica(cleaned):
        return {'text': cleaned, 'format': 'auto_storica'}

    sub = find_pattern_substring(cleaned, 7, looks_like_moto)
    if sub:
        return {'text': sub, 'format': 'moto'}

    sub = find_pattern_substring(cleaned, 8, looks_like_storica)
    if sub:
        return {'text': sub, 'format': 'auto_storica'}

    best = sliding_window_best(cleaned)
    if is_auto_moderna(best):
        return {'text': best, 'format': 'auto_moderna'}
    return {'text': best, 'format': 'unknown'}


if __name__ == "__main__":
    tests = [
        # Auto moderne pulite
        ("CM128ZK",            "CM128ZK",  "auto_moderna"),
        ("DW397AK",            "DW397AK",  "auto_moderna"),
        ("AE299YT",            "AE299YT",  "auto_moderna"),
        # IT iniziale
        ("ITCM128ZK",          "CM128ZK",  "auto_moderna"),
        ("CM128ZKVA",          "CM128ZK",  "auto_moderna"),
        ("ITCM128ZKVA",        "CM128ZK",  "auto_moderna"),
        # Correzione I/O/Q nelle cifre
        ("CMI28ZK",            "CM128ZK",  "auto_moderna"),    # I->1 in pos 2
        ("CM12OZK",            "CM120ZK",  "auto_moderna"),    # O->0 in pos 4
        ("CM1Q8ZK",            "CM108ZK",  "auto_moderna"),    # Q->0 in pos 3
        ("CMIOQZK",            "CM100ZK",  "auto_moderna"),    # tutti e 3
        ("ITCMI28ZKVA",        "CM128ZK",  "auto_moderna"),    # combinato
        # Moto
        ("AB12345",            "AB12345",  "moto"),
        ("XY99999",            "XY99999",  "moto"),
        # Storiche
        ("NO772858",           "NO772858", "auto_storica"),
        ("MI123456",           "MI123456", "auto_storica"),
        # Rumorose
        ("IRLBKMOTORSDW397AK", "DW397AK",  "auto_moderna"),
        # Casi limite
        ("",                   "",         "unknown"),
    ]
    print(f"{'Input':<28} {'Expected':<15} {'Got':<15} {'Format':<15} {'OK'}")
    print("-" * 85)
    n_ok = 0
    for raw, exp_t, exp_f in tests:
        r = postprocess_plate(raw)
        ok = (r['text'] == exp_t and r['format'] == exp_f)
        n_ok += int(ok)
        print(f"{raw:<28} {exp_t:<15} {r['text']:<15} {r['format']:<15} {'OK' if ok else 'FAIL'}")
    print("-" * 85)
    print(f"Passed: {n_ok}/{len(tests)}")
