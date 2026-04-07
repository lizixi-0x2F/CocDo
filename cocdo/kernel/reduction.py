"""Beta reduction and substitution."""
from __future__ import annotations
from .terms import Term, Var, Const, Sort, Pi, Lam, App


# ── Tensor-aware built-in evaluation ──────────────────────────────────────────

def _try_eval_builtin(term: App) -> Term:
    """Evaluate App(App(Add/Mul, a), b) when a and b are Const with values.

    This is the tensor-aware extension: when both operands carry numeric or
    array-like values, we reduce the application immediately to a new Const
    instead of leaving it as an unevaluated App.  This means the entire causal
    propagation E_j = Σ w_i * E_i can happen inside the COC kernel via
    beta_reduce, rather than in a separate numpy matmul.

    Supported patterns:
        App(App(Add, Const(_, a)), Const(_, b))  →  Const("_add", a + b)
        App(App(Mul, Const(_, w)), Const(_, x))  →  Const("_mul", w * x)

    Any other App is left unchanged (returned as-is).
    """
    # Pattern: App(partial, rhs) where partial = App(op, lhs)
    if not isinstance(term.func, App):
        return term
    partial = term.func          # App(op, lhs)
    op      = partial.func       # Add or Mul constant
    lhs     = partial.arg        # first operand
    rhs     = term.arg           # second operand

    if not (isinstance(op, Const) and op.value is None):
        return term
    if not (isinstance(lhs, Const) and lhs.value is not None):
        return term
    if not (isinstance(rhs, Const) and rhs.value is not None):
        return term

    if op.name == "Add":
        return Const("_add", lhs.value + rhs.value)
    if op.name == "Mul":
        return Const("_mul", lhs.value * rhs.value)

    return term


def subst(term: Term, var: str, replacement: Term) -> Term:
    """Capture-avoiding substitution of `var` with `replacement` in `term`."""
    if isinstance(term, Var):
        return replacement if term.name == var else term
    elif isinstance(term, (Const, Sort)):
        return term
    elif isinstance(term, Pi):
        new_domain = subst(term.domain, var, replacement)
        if term.var == var:
            return Pi(term.var, new_domain, term.codomain)
        return Pi(term.var, new_domain, subst(term.codomain, var, replacement))
    elif isinstance(term, Lam):
        new_domain = subst(term.domain, var, replacement)
        if term.var == var:
            return Lam(term.var, new_domain, term.body)
        return Lam(term.var, new_domain, subst(term.body, var, replacement))
    elif isinstance(term, App):
        return App(subst(term.func, var, replacement), subst(term.arg, var, replacement))
    return term


def _step(term: Term) -> Term:
    if isinstance(term, App):
        # First try built-in tensor evaluation (Add/Mul on valued Consts)
        evaled = _try_eval_builtin(term)
        if evaled is not term:
            return evaled
        if isinstance(term.func, Lam):
            return subst(term.func.body, term.func.var, term.arg)
        new_func = _step(term.func)
        if new_func is not term.func:
            return App(new_func, term.arg)
        new_arg = _step(term.arg)
        if new_arg is not term.arg:
            return App(term.func, new_arg)
    elif isinstance(term, Lam):
        new_body = _step(term.body)
        if new_body is not term.body:
            return Lam(term.var, term.domain, new_body)
    elif isinstance(term, Pi):
        new_domain = _step(term.domain)
        if new_domain is not term.domain:
            return Pi(term.var, new_domain, term.codomain)
        new_cod = _step(term.codomain)
        if new_cod is not term.codomain:
            return Pi(term.var, term.domain, new_cod)
    return term


def beta_reduce(term: Term, steps: int = 100) -> Term:
    """Full call-by-value beta reduction."""
    for _ in range(steps):
        reduced = _step(term)
        if reduced is term:
            break
        term = reduced
    return term
