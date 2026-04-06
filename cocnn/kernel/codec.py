"""COC Term <-> token sequence codec for neural generation."""
from __future__ import annotations
from typing import Optional
from .terms import Term, Sort, Var, Const, Pi, Lam, App


# ── Vocabulary ────────────────────────────────────────────────────────────────

# Structural tokens
SORT = "SORT"
_VAR = "VAR"       # underscore to avoid shadowing terms.Var
_CONST = "CONST"
_PI = "PI"
_LAM = "LAM"
_APP = "APP"
SEP = "SEP"
END = "END"
EOS = "EOS"
PAD = "PAD"

# Variable name pool
VAR_NAMES = [chr(c) for c in range(ord("a"), ord("z") + 1)] + \
            [chr(c) for c in range(ord("A"), ord("Z") + 1)]

# Sort level tokens
LEVEL_TOKENS = [f"L{i}" for i in range(4)]   # L0, L1, L2, L3

STRUCTURAL_TOKENS = [PAD, SORT, _VAR, _CONST, _PI, _LAM, _APP, SEP, END, EOS]
ALL_TOKENS = STRUCTURAL_TOKENS + VAR_NAMES + LEVEL_TOKENS

TOKEN_TO_ID = {tok: i for i, tok in enumerate(ALL_TOKENS)}
ID_TO_TOKEN = {i: tok for tok, i in TOKEN_TO_ID.items()}
VOCAB_SIZE = len(ALL_TOKENS)

PAD_ID = TOKEN_TO_ID[PAD]
EOS_ID = TOKEN_TO_ID[EOS]


# ── Encode: Term → token list ────────────────────────────────────────────────

def encode(term: Term) -> list[str]:
    """Serialize a COC Term to a token sequence (pre-order traversal)."""
    if isinstance(term, Sort):
        level = min(term.level, 3)
        return [SORT, f"L{level}"]

    elif isinstance(term, Var):
        name = term.name if term.name in VAR_NAMES else "x"
        return [_VAR, name]

    elif isinstance(term, Const):
        name = term.name if term.name in VAR_NAMES else "c"
        return [_CONST, name]

    elif isinstance(term, Pi):
        var = term.var if term.var in VAR_NAMES else "x"
        return [_PI, var] + encode(term.domain) + [SEP] + encode(term.codomain) + [END]

    elif isinstance(term, Lam):
        var = term.var if term.var in VAR_NAMES else "x"
        return [_LAM, var] + encode(term.domain) + [SEP] + encode(term.body) + [END]

    elif isinstance(term, App):
        return [_APP] + encode(term.func) + [SEP] + encode(term.arg) + [END]

    raise ValueError(f"Unknown term type: {type(term)}")


def encode_to_ids(term: Term) -> list[int]:
    """Serialize a COC Term to a list of token IDs."""
    return [TOKEN_TO_ID[t] for t in encode(term)] + [EOS_ID]


# ── Decode: token list → Term ────────────────────────────────────────────────

class DecodeError(Exception):
    pass


class _Decoder:
    """Recursive descent parser for COC token sequences."""

    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[str]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self) -> str:
        if self.pos >= len(self.tokens):
            raise DecodeError("Unexpected end of token sequence")
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, expected: str):
        tok = self.consume()
        if tok != expected:
            raise DecodeError(f"Expected {expected!r}, got {tok!r} at pos {self.pos - 1}")

    def parse_term(self) -> Term:
        tok = self.consume()

        if tok == SORT:
            level_tok = self.consume()
            if level_tok.startswith("L") and level_tok[1:].isdigit():
                return Sort(int(level_tok[1:]))
            raise DecodeError(f"Expected level token, got {level_tok!r}")

        elif tok == _VAR:
            name = self.consume()
            return Var(name)

        elif tok == _CONST:
            name = self.consume()
            return Const(name)

        elif tok == _PI:
            var = self.consume()
            domain = self.parse_term()
            self.expect(SEP)
            codomain = self.parse_term()
            self.expect(END)
            return Pi(var, domain, codomain)

        elif tok == _LAM:
            var = self.consume()
            domain = self.parse_term()
            self.expect(SEP)
            body = self.parse_term()
            self.expect(END)
            return Lam(var, domain, body)

        elif tok == _APP:
            func = self.parse_term()
            self.expect(SEP)
            arg = self.parse_term()
            self.expect(END)
            return App(func, arg)

        elif tok == EOS:
            raise DecodeError("Unexpected EOS")

        raise DecodeError(f"Unexpected token {tok!r} at pos {self.pos - 1}")


def decode(tokens: list[str]) -> Term:
    """Parse a token sequence back into a COC Term."""
    # Strip trailing EOS/PAD
    clean = [t for t in tokens if t not in (EOS, PAD)]
    if not clean:
        raise DecodeError("Empty token sequence")
    parser = _Decoder(clean)
    term = parser.parse_term()
    return term


def decode_from_ids(ids: list[int]) -> Term:
    """Parse a list of token IDs back into a COC Term."""
    tokens = []
    for i in ids:
        tok = ID_TO_TOKEN.get(i)
        if tok is None:
            raise DecodeError(f"Unknown token ID: {i}")
        if tok == EOS:
            break
        if tok == PAD:
            continue
        tokens.append(tok)
    return decode(tokens)


# ── Utility ───────────────────────────────────────────────────────────────────

def term_depth(term: Term) -> int:
    """Return the AST depth of a COC Term."""
    if isinstance(term, (Sort, Var, Const)):
        return 1
    elif isinstance(term, Pi):
        return 1 + max(term_depth(term.domain), term_depth(term.codomain))
    elif isinstance(term, Lam):
        return 1 + max(term_depth(term.domain), term_depth(term.body))
    elif isinstance(term, App):
        return 1 + max(term_depth(term.func), term_depth(term.arg))
    return 1
