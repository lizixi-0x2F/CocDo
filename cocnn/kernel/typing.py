"""COC type checker — validates causal structure consistency."""
from __future__ import annotations
from typing import Dict
from .terms import Term, Sort, Var, Const, Pi, Lam, App
from .reduction import beta_reduce, subst

Context = Dict[str, Term]


def type_of(term: Term, ctx: Context) -> Term:
    """Return the type of `term` under `ctx`. Raises TypeError on failure."""
    if isinstance(term, Sort):
        return Sort(term.level + 1)

    elif isinstance(term, Var):
        if term.name not in ctx:
            raise TypeError(f"Free variable {term.name!r} not in context")
        return ctx[term.name]

    elif isinstance(term, Const):
        if term.name not in ctx:
            raise TypeError(f"Constant {term.name!r} has no declared type")
        return ctx[term.name]

    elif isinstance(term, Pi):
        A_type = type_of(term.domain, ctx)
        new_ctx = {**ctx, term.var: term.domain}
        B_type = type_of(term.codomain, new_ctx)
        if isinstance(A_type, Sort) and isinstance(B_type, Sort):
            return Sort(max(A_type.level, B_type.level))
        raise TypeError(f"Malformed Pi type: {term}")

    elif isinstance(term, Lam):
        new_ctx = {**ctx, term.var: term.domain}
        body_type = type_of(term.body, new_ctx)
        return Pi(term.var, term.domain, body_type)

    elif isinstance(term, App):
        func_type = beta_reduce(type_of(term.func, ctx))
        if not isinstance(func_type, Pi):
            raise TypeError(f"Applied non-function type: {func_type}")
        return beta_reduce(subst(func_type.codomain, func_type.var, term.arg))

    raise TypeError(f"Unknown term: {term}")
