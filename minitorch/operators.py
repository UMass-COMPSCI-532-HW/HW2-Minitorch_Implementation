"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
def mul(x: float, y: float) -> float:
    """$f(x, y) = x * y$"""
    return x * y
# TODO fill in the remaining mathematical operators. Use mul provided above as an example.
# - id
def id(x: float) -> float:
    """$f(x) = x$"""
    return x
# - add
def add(x: float, y: float) -> float:
    """$f(x, y) = x + y$"""
    return x + y
# - addLists
def addLists(xs: Iterable[float], ys: Iterable[float]) -> list[float]:
    """$f(x, y) = x + y$ for each corresponding element in the lists"""
    return [x + y for x, y in zip(xs, ys)]
# - neg
def neg(x: float) -> float:
    """$f(x) = -x$"""
    return -x
# - negList
def negList(xs: Iterable[float]) -> list[float]:
    """$f(x) = -x$ for each element in the list"""
    return [-x for x in xs]
# - prod
def prod(xs: Iterable[float]) -> float:
    """$f(x) = \prod_{i} x_i$"""
    result = 1.0
    for x in xs:
        result *= x
    return result
# - sum
def sum(xs: Iterable[float]) -> float:
    """$f(x) =\ sum_{i} x_i$"""
    result = 0.0
    for x in xs:
        result += x
    return result
# - lt
def lt(x: float, y: float) -> float:
    """$f(x) = 1.0 if x < y else 0.0$"""
    return 1.0 if x < y else 0.0
# - eq
def eq(x: float, y: float) -> float:
    """$f(x) = 1.0 if x == y else 0.0$"""
    return 1.0 if x == y else 0.0
# - max
def max(x: float, y: float) -> float:
    """$f(x, y) = x$ if x > y else $f(x, y) = y$"""
    return x if x > y else y
# - is_close
def is_close(x: float, y: float) -> float:
    """$f(x, y) = 1.0$ if $|x - y| < 1e-2$ else $f(x, y) = 0.0$"""
    return 1.0 if abs(x - y) < 1e-2 else 0.0
# - sigmoid
def sigmoid(x: float) -> float:
    """$f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$"""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
# - relu
def relu(x: float) -> float:
    """$f(x) = x$ if x > 0, else $f(x) = 0$"""
    return x if x > 0 else 0
# - log
def log(x: float) -> float:
    """$f(x) = log(x)$"""
    return math.log(x)
# - exp
def exp(x: float) -> float:
    """$f(x) = e^x$"""
    return math.exp(x)
# - log_back
def log_back(x: float, d: float) -> float:
    """$f(x) = log_e(x)$ if x > 0 else $f(x) = 0$"""
    return d / x if x > 0 else 0
# - inv
def inv(x: float) -> float:
    """$f(x) = 1/x$"""
    return 1 / x
# - inv_back
def inv_back(x: float, d: float) -> float:
    """$f(x) = 1/x$ if x > 0 else $f(x) = 0$"""
    return -d / x**2 if x > 0 else 0
# - relu_back
def relu_back(x: float, d: float) -> float:
    """$f(x) = 1.0$ if x > 0 else $f(x) = 0.0$"""
    return d if x > 0 else 0
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
def sigmoid_back(x: float, d: float) -> float:
    """$f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$"""
    sig = sigmoid(x)
    return d * sig * (1 - sig) if x >= 0 else d * (math.exp(x) / (1 + math.exp(x))**2)

# For is_close:
# $f(x) = |x - y| < 1e-2$
def is_close_back(x: float, y: float, d: float) -> tuple[float, float]:
    """$f(x, y) = 1.0$ if $|x - y| < 1e-2$ else $f(x, y) = 0.0$"""
    return (d if abs(x - y) < 1e-2 else 0.0, d if abs(x - y) < 1e-2 else 0.0)





# TODO: Implement for Task 0.1.
EPS = 1e-6


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate all elemnts in a list using map
# - addLists : add corresponding elements from two lists using zipWith
# - sum: sum all elements in a list using reduce
# - prod: tcalculate the product of all elements in a list using reduce


# TODO: Implement for Task 0.3.
