"""Collection of the core mathematical operators used throughout the code base."""

import math
import numpy as np
from numpy import isclose

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
def mul(x: float, y: float) -> float:
    """$f(x, y) = x * y$"""
    return np.multiply(x, y)
# TODO fill in the remaining mathematical operators. Use mul provided above as an example.
# - id
def id(x: float) -> float:
    """$f(x) = x$"""
    return np.copy(x).item()
# - add
def add(x: float, y: float) -> float:
    """$f(x, y) = x + y$"""
    return np.add(x, y)
# # - addLists
# def addLists(xs: Iterable[float], ys: Iterable[float]) -> list[float]:
#     """$f(x, y) = x + y$ for each corresponding element in the lists"""
#     arr_xs = np.array(xs)
#     arr_ys = np.array(ys)
#     return np.add(arr_xs, arr_ys).tolist()
# - neg
def neg(x: float) -> float:
    """$f(x) = -x$"""
    return -x
# - negList
# def negList(xs: Iterable[float]) -> list[float]:
#     """$f(x) = -x$ for each element in the list"""
#     arr = np.array(xs)
#     return np.negative(arr).tolist()
# - prod
# def prod(xs: Iterable[float]) -> float:
#     """$f(x) = prod_{i} x_i$"""
#     arr = np.array(xs)
#     return np.prod(arr).item()
# - sum
# def sum(xs: Iterable[float]) -> float:
#     """$f(x) = sum_{i} x_i$"""
#     arr = np.array(xs)
#     return np.sum(arr).item()
# - lt
def lt(x: float, y: float) -> float:
    """$f(x) = 1.0 if x < y else 0.0$"""
    return 1.0 if np.less(x, y) else 0.0
# - eq
def eq(x: float, y: float) -> float:
    """$f(x) = 1.0 if x == y else 0.0$"""
    return 1.0 if np.equal(x, y) else 0.0
# - max
def max(x: float, y: float) -> float:
    """$f(x, y) = x$ if x > y else $f(x, y) = y$"""
    return np.maximum(x, y).item()
# - is_close
def is_close(x: float, y: float) -> float:
    """$f(x, y) = 1.0$ if $|x - y| < 1e-2$ else $f(x, y) = 0.0$"""
    return 1.0 if np.isclose(x, y, atol=1e-2) else 0.0
# - sigmoid
def sigmoid(x: float) -> float:
    """$f(x) =  frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$""" 
    if x >= 0:
        return np.divide(1.0, np.add(1.0, np.exp(-x))).item()
    else:
        return np.divide(np.exp(x), np.add(1.0, np.exp(x))).item()
    

# - relu
def relu(x: float) -> float:
    """$f(x) = x$ if x > 0, else $f(x) = 0$"""
    return np.maximum(x, 0).item()
# - log
def log(x: float) -> float:
    """$f(x) = log(x)$"""
    return np.log(x).item()
# - exp
def exp(x: float) -> float:
    """$f(x) = e^x$"""
    return np.exp(x).item()
# - log_back
def log_back(x: float, d: float) -> float:
    """$f(x) = log_e(x)$ if x > 0 else $f(x) = 0$"""
    return np.divide(d, x) if x > 0 else 0.0
# - inv
def inv(x: float) -> float:
    """$f(x) = 1/x$"""
    return np.reciprocal(x).item()
# - inv_back
def inv_back(x: float, d: float) -> float:
    """$f(x) = 1/x$ if x > 0 else $f(x) = 0$"""
    return -np.divide(d, x**2) if x > 0 else 0.0
# - relu_back
def relu_back(x: float, d: float) -> float:
    """$f(x) = 1.0$ if x > 0 else $f(x) = 0.0$"""
    return d if np.greater(x, 0) else 0.0

# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
def sigmoid_back(x: float, d: float) -> float:
    """$f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$"""
    sig = sigmoid(x)
    return np.multiply(d, np.multiply(sig, np.subtract(1, 
        sig))) if x >= 0 else np.multiply(d, np.divide(math.exp(x), 
                                        (1 + math.exp(x))**2))


# For is_close:
# $f(x) = |x - y| < 1e-2$
def is_close_back(x: float, y: float, d: float) -> tuple[float, float]:
    """$f(x, y) = 1.0$ if $|x - y| < 1e-2$ else $f(x, y) = 0.0$"""
    return (d if np.isclose(x, y, atol=1e-2) else 0.0, 
            d if np.isclose(x, y, atol=1e-2) else 0.0)





# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
def map(fn: Callable[[float], float], xs: Iterable[float]) -> list[float]:
    """$f(x) = fn(x)$ for each element in the list"""
    arr = np.array(xs)
    if arr.size == 0:
        return []
    return np.vectorize(fn)(arr).tolist()

# - zipWith
def zipWith(fn: Callable[[float, float], float], xs: Iterable[float], ys: Iterable[float]) -> list[float]:
    """$f(x, y) = fn(x, y)$ for each corresponding element in the lists"""
    arr_xs = np.array(xs)
    arr_ys = np.array(ys)
    if arr_xs.size == 0 or arr_ys.size == 0:
        return []
    return np.vectorize(fn)(arr_xs, arr_ys).tolist()

# - reduce
def reduce(fn: Callable[[float, float], float], xs: Iterable[float], init: float) -> float:
    """$f(x, y) = fn(x, y)$ for each element in the list"""
    arr = np.array(xs)
    if arr.size == 0:
        return init
    return float(np.frompyfunc(fn, 2, 1).reduce(arr, initial=init))

#
# Use these to implement
# - negList : negate all elemnts in a list using map
def negList(xs: Iterable[float]) -> list[float]:
    """$f(x) = -x$ for each element in the list"""
    return map(neg, xs)
# - addLists : add corresponding elements from two lists using zipWith
def addLists(xs: Iterable[float], ys: Iterable[float]) -> list[float]:
    """$f(x, y) = x + y$ for each corresponding element in the lists"""
    return zipWith(add, xs, ys)
# - sum: sum all elements in a list using reduce
def sum(xs: Iterable[float]) -> float:
    """$f(x) = sum_{i} x_i$"""
    return reduce(add, xs, 0.0)
# - prod: tcalculate the product of all elements in a list using reduce
def prod(xs: Iterable[float]) -> float:
    """$f(x) = prod_{i} x_i$"""
    return reduce(mul, xs, 1.0)


# TODO: Implement for Task 0.3.
