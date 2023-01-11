import inspect
from collections.abc import Iterable


def make_iterable(x):
    if isinstance(x, Iterable):
        return x
    else:
        return [x]


def check_mechanism_signature(fct: callable, parents: list[int]) -> bool:
    n_parents = (len(parents) if parents is not None else 0)
    if len(inspect.signature(fct).parameters) == n_parents + 1:
        return True
    else:
        return False
