import z3

from .util import *

# This file contains global symbols for Intel pseudocode.
# Basically, these are implementations of built-in functions that Intel doesn't
# feel the need to specify apparently.

def POPCNT(bits, **kwargs):
    # This is a variable-width version of the classic 0x5555/0x3333/0x0f0f/0xffff
    # etc algorithm, to sum N bits in O(log2 N) steps
    shift = 1
    while shift < bits.size():
        mask = sum(1 << x for x in range(bits.size()) if not x & shift)
        bits = (bits & mask) + ((bits >> shift) & mask)
        shift *= 2
    return bits & ((1 << shift) - 1)

# Create z3 lambdas for comparison operations
_a, _b = z3.Ints('a b')

_MM_CMPINT_EQ     = z3.Lambda([_a, _b], _a == _b)           # 0
_MM_CMPINT_LT     = z3.Lambda([_a, _b], _a < _b)            # 1
_MM_CMPINT_LE     = z3.Lambda([_a, _b], _a <= _b)           # 2
_MM_CMPINT_FALSE  = z3.Lambda([_a, _b], z3.BoolVal(False))  # 3
_MM_CMPINT_NE     = z3.Lambda([_a, _b], _a != _b)           # 4
_MM_CMPINT_NLT    = z3.Lambda([_a, _b], _a >= _b)           # 5
_MM_CMPINT_NLE    = z3.Lambda([_a, _b], _a > _b)            # 6
_MM_CMPINT_TRUE   = z3.Lambda([_a, _b], z3.BoolVal(True))   # 7

# Zero/sign extension: just mark these values as signed/unsigned, the users
# of the values will use this later when the size of the target value is known

def ZeroExtend(v, **kwargs):
    return Value(v, signed=False)

ZeroExtend16 = ZeroExtend

def SignExtend(v, **kwargs):
    return Value(v, signed=True)

Signed = SignExtend

# XXX Intel uses the same freakin functions for signed and unsigned comparison...

def _cmp(cmp, a, b, **kwargs):
    [a, b, width, signed] = match_types(a, b)
    a = z3.BV2Int(a, is_signed=signed)
    b = z3.BV2Int(b, is_signed=signed)
    return z3.Int2BV(z3.If(cmp(a, b), a, b), width)

def MIN(a, b, **kwargs):
    return _cmp(lambda a, b: a < b, a, b, **kwargs)

def MAX(a, b, **kwargs):
    return _cmp(lambda a, b: a > b, a, b, **kwargs)
