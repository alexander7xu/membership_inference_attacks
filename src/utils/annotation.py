from beartype import beartype as typechecked
from jaxtyping import Float as FP
from jaxtyping import Int, jaxtyped
from torch import Tensor as T

__all__ = ["FP", "Int", "T", "tensor_typechecked", "typechecked"]


def tensor_typechecked(func):
    return jaxtyped(func, typechecker=typechecked)
