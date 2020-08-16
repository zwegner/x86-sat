import z3

def is_z3(v):
    return z3.is_expr(v) or z3.is_sort(v)

class Value:
    def __init__(self, value, signed=None, width=None):
        if isinstance(value, Value):
            if signed is None:
                signed = value.signed
            if width is None:
                width = value.width
            value = value.value
        if not z3.is_expr(value) and width is not None:
            # XXX range check
            value = z3.BitVecVal(value, width)
        self.value = value
        self.signed = signed
        self.width = width

    def size(self):
        return self.width

    def __eq__(self, other):
        return self.value == other

def try_simplify(v):
    if isinstance(v, Value):
        return Value(try_simplify(v.value), signed=v.signed, width=v.width)
    if is_z3(v):
        return z3.simplify(v)
    return v

# Extend a value to a given number of bits, either sign- or zero-extended based
# on the signedness of the value
def extend(value, width, signed=None):
    if not is_z3(value):
        return value
    # XXX range check
    if not z3.is_bv(value):
        value = z3.Int2BV(value, width)
    assert value.size() > 0
    diff = try_simplify(width - value.size())
    if diff > 0:
        if signed is None:
            # XXX default to unsigned
            signed = False
        if signed:
            return z3.SignExt(diff, value)
        else:
            return z3.ZeroExt(diff, value)
    return value

def match_types(lhs, rhs):
    # Get width/signedness of each arg
    widths = []
    signs = []
    values = []
    for value in [lhs, rhs]:
        value = try_simplify(value)

        signed = None
        width = None
        if is_z3(value):
            width = value.size()
        elif isinstance(value, Value):
            signed = value.signed
            width = value.width
            value = value.value

        if width is not None:
            widths.append(width)
        signs.append(signed)
        values.append(value)
    [lhs, rhs] = values

    # Check for a single signedness, in which case the result can take that
    # signedness. Otherwise, each value gets extended on its own and the
    # result is ambiguous
    signed = None
    signs_set = set(signs) - {None}
    if len(signs_set) == 1:
        [signed] = signs_set
        signs = [signed, signed]

    # Check if either arg has an explicit width, and take the max if both do,
    # then extend both args based on their signedness
    width = None
    if widths:
        width = max(widths)
        lhs = extend(lhs, width, signed=signs[0])
        rhs = extend(rhs, width, signed=signs[1])

    return [lhs, rhs, width, signed]
