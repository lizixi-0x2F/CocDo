"""COC AST nodes — no torch dependency."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union


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
    """Atomic constant — carries a continuous value from the neural network."""
    name: str
    value: Optional[float] = None
    def __repr__(self):
        return f"<{self.name}={self.value:.3f}>" if self.value is not None else f"<{self.name}>"


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
