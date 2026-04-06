"""Beta reduction and substitution."""
from __future__ import annotations
from .terms import Term, Var, Const, Sort, Pi, Lam, App


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
