from torch import Tensor as T
from jaxtyping import Float as FP
from jaxtyping import Int, jaxtyped
from typeguard import typechecked


def tensor_typechecked(func):
    return jaxtyped(func, typechecker=typechecked)
