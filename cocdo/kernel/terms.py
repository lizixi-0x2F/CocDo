"""COC AST nodes — no torch dependency."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union, Any


class Sort:
    """Universe levels: Prop and Type_i."""
    def __init__(self, level: int = 0, is_prop: bool = False):
        self.level = level
        self.is_prop = is_prop

    def __repr__(self):
        return "Prop" if self.is_prop else f"Type{self.level}"

    def __eq__(self, other):
        return isinstance(other, Sort) and self.level == other.level and self.is_prop == other.is_prop


Term = Union["Var", "Sort", "Pi", "Lam", "App", "Const"]


@dataclass
class Var:
    name: str
    def __repr__(self): return self.name


@dataclass
class Const:
    """Atomic constant — optionally carries a semantic embedding (any array-like).

    `value` can be:
      - None          : pure symbol
      - float         : scalar (legacy causal weight)
      - torch.Tensor  : semantic embedding vector from MiniLM
      - np.ndarray    : same, numpy form
    """
    name:  str
    value: Any = field(default=None, compare=False)

    def __repr__(self):
        if self.value is None:
            return f"<{self.name}>"
        try:
            # scalar
            return f"<{self.name}={float(self.value):.3f}>"
        except (TypeError, ValueError):
            # tensor/array — show shape
            try:
                shape = tuple(self.value.shape)
                return f"<{self.name}:emb{shape}>"
            except AttributeError:
                return f"<{self.name}:?>"


@dataclass
class Pi:
    """forall(x:A).B — type-theoretic encoding of a causal dependency."""
    var: str
    domain: Term
    codomain: Term
    def __repr__(self): return f"(Pi {self.var}:{self.domain}. {self.codomain})"


@dataclass
class Lam:
    """Lambda abstraction — implementation of a causal mechanism."""
    var: str
    domain: Term
    body: Term
    def __repr__(self): return f"(lam {self.var}:{self.domain}. {self.body})"


@dataclass
class App:
    func: Term
    arg: Term
    def __repr__(self): return f"({self.func} {self.arg})"


# ── Built-in constants ─────────────────────────────────────────────────────────

# Binary addition:       App(App(Add, a), b)  →  a + b
# Scalar multiplication: App(App(Mul, w), x)  →  w * x
# Used to combine weighted parent contributions in multi-parent mechanisms.
Add = Const("Add")
Mul = Const("Mul")
