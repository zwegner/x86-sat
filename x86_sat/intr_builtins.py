import z3

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

# Should maybe handle this better...
def ZeroExtend(v, **kwargs):
    return v
