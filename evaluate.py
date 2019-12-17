import contextlib

import z3

# Globals symbols for Intel pseudocode
INTR_GLOBALS = {
    # Should maybe handle this better...
    'ZeroExtend': lambda v, **kwargs: v,
}

# Context handles the current symbol values during execution, and predication.
# Predication is required for handling branches on unknown data, which get
# transformed into a Z3 If() over the results of both branches.
class Context:
    def __init__(self, pred=None):
        self.symbols = {}
        self.pred = pred
    def set(self, name, value):
        self.symbols[name] = value
    def get(self, name):
        if name in self.symbols:
            return self.symbols[name]
        return INTR_GLOBALS.get(name)

    # If a predicate is active, make an expression conditional
    def predicate(self, true_expr, false_expr):
        if self.pred is not None:
            return z3.If(self.pred, true_expr, false_expr)
        return true_expr

    # Run a block of code under a given predicate
    @contextlib.contextmanager
    def predicated(self, expr):
        old_pred = self.pred
        # AND with the old predicate if there is one
        if old_pred is not None:
            expr = z3.And(old_pred, expr)

        self.pred = expr
        yield
        self.pred = old_pred

def is_z3(v):
    return z3.is_expr(v) or z3.is_sort(v)

# Weird hacky functions to "satisfy" Z3 (get it?)
# We need to make sure we're dealing with the right bit-vector widths, etc.,
# and not pollute the code with messy Z3 shit everywhere, so these are some
# generic functions to munge things around.

def try_simplify(v):
    if is_z3(v):
        return z3.simplify(v)
    return v

def try_bool(b):
    b = try_simplify(b)
    # HACK: z3 python interface has bug/weird behavior where (x == y) is
    # always False for unknown x and y, so use is_true and is_false instead
    if z3.is_true(b):
        return True
    if z3.is_false(b):
        return False
    return None

def zero_ext(value, width):
    if not is_z3(value):
        return value
    assert value.size() > 0
    diff = try_simplify(width - value.size())
    if diff > 0:
        return z3.ZeroExt(diff, value)
    return value

# Make sure two operands are the same width by zero-extending the smaller one.
def match_width(lhs, rhs):
    if is_z3(lhs) or is_z3(rhs):
        width = max([v.size() for v in [lhs, rhs] if is_z3(v)])
        return [zero_ext(lhs, width), zero_ext(rhs, width)]
    return [lhs, rhs]

# The "add" argument adds more bits, and is needed at least by left shift.
# Intel's code uses stuff like (bit << 2), which needs to be 3 bits, not 1.
# "double" doubles the width. This is needed for multiplications, which
# could silently overflow before.
def match_width_fn(lhs, rhs, fn, add=0, double=False):
    if is_z3(lhs) or is_z3(rhs):
        width = max([v.size() for v in [lhs, rhs] if is_z3(v)]) + add
        if double:
            width *= 2
        return fn(zero_ext(lhs, width), zero_ext(rhs, width))
    return fn(lhs, rhs)

# For pretty printing
def indent(s):
    return '    ' + str(s).replace('\n', '\n    ')

# Shortcut for making a Z3 bitvector with the right number of bits for a C type
def var(name, type):
    width = {
        'char': 8,
        '__mmask8': 8,

        'short': 16,
        '__mmask16': 16,

        'int': 32,
        'const int': 32,
        '__mmask32': 32,

        'long long': 64,
        '__int64': 64,
        '__mmask64': 64,

        '__m64': 64,
        '__m128i': 128,
        '__m256': 256,
        '__m256d': 256,
        '__m256i': 256,
        '__m512i': 512,
    }[type]
    return z3.BitVec(name, width)

# AST types. These handle all evaluation

class Node:
    pass

# Decorator for easily making Node subclasses with given parameters
def node(*params):
    def decorate(cls):
        # Redefine class to have Node as parent. Cool hack bro.
        cls = type(cls.__name__, (Node,), dict(cls.__dict__))

        def __init__(self, *args, info=None):
            assert len(args) == len(params)
            for p, a in zip(params, args):
                setattr(self, p, a)
            self.info = info or args[0]

        def __eq__(self, other):
            if type(self) != type(other): return False
            return all(getattr(self, a) == getattr(other, a) for a in params)

        cls.__init__ = __init__
        cls.__eq__ = __eq__
        return cls
    return decorate

@node('name')
class Identifier:
    def eval(self, ctx):
        return ctx.get(self.name)
    def __repr__(self):
        return self.name

@node('value')
class Integer:
    def eval(self, ctx):
        return self.value
    def __repr__(self):
        return '%s' % self.value

@node('op', 'rhs')
class UnaryOp:
    def eval(self, ctx):
        if self.op == 'NOT':
            return ~self.rhs.eval(ctx)
        assert False
    def __repr__(self):
        return '(%s %s)' % (self.op, self.rhs)

@node('op', 'lhs', 'rhs')
class BinaryOp:
    def eval(self, ctx):
        lhs, rhs = match_width(self.lhs.eval(ctx), self.rhs.eval(ctx))
        if self.op == '+':
            return lhs + rhs
        elif self.op == '-':
            return lhs - rhs
        elif self.op == '*':
            return match_width_fn(lhs, rhs, lambda l, r: l * r, double=True)
        elif self.op == 'AND':
            return lhs & rhs
        elif self.op == 'OR':
            return lhs | rhs
        elif self.op == 'XOR':
            return lhs ^ rhs
        elif self.op == '<<':
            # Add more bits to the left if we know the rhs
            add = rhs if isinstance(rhs, int) else 0
            return match_width_fn(lhs, rhs, lambda l, r: l << r, add=add)
        elif self.op == '>>':
            return lhs >> rhs
        elif self.op == '<':
            return lhs < rhs
        elif self.op == '>':
            return lhs > rhs
        elif self.op == '==':
            return lhs == rhs
        assert False, 'unknown binop %s' % self.op
    def __repr__(self):
        return '(%s %s %s)' % (self.lhs, self.op, self.rhs)

@node('expr', 'hi', 'lo')
class Slice:
    def eval(self, ctx):
        expr = self.expr.eval(ctx)
        hi, lo = self.hi.eval(ctx), self.lo.eval(ctx)
        width = hi - lo + 1

        if is_z3(expr):
            shifted = match_width_fn(expr, lo, lambda l, r: l >> r)

            # Big hack! Simplify (x+y)-x -> y to get the width when we don't
            # know x. This is pretty common, e.g. a[index*8+7:index*8]
            if (isinstance(self.hi, BinaryOp) and self.hi.op == '+' and
                    self.hi.lhs == self.lo):
                width1 = try_simplify(self.hi.rhs.eval(ctx))
                return z3.Extract(width1, 0, shifted)

            # HACK!! Z3 will complain if we try to Extract with a non-constant
            # range [lo, hi], since it can't determine if lo <= hi. Well, it
            # actually can, but the Z3 python wrapper doesn't simplify that
            # expression before checking it. So, as a special case, see if we
            # can at least determine that lo == hi (which is common because Slice
            # is used for single-bit indexing like a[i]), then replace the
            # expression with a variable shift and a single-bit extract.
            if is_z3(hi) and is_z3(lo) and try_bool(hi == lo):
                return z3.Extract(0, 0, shifted)

            return z3.Extract(width - 1, 0, shifted)

        # Slice integers with normal bit ops
        assert width > 0
        mask = ((1 << width) - 1)
        return (expr >> lo) & mask

    def __repr__(self):
        return '%s[%s:%s]' % (self.expr, self.hi, self.lo)

@node('fn', 'args')
class Call:
    def eval(self, ctx):
        fn = self.fn.eval(ctx)
        assert callable(fn)
        args = [a.eval(ctx) for a in self.args]
        return fn(*args, pred=ctx.pred)
    def __repr__(self):
        return '%s(%s)' % (self.fn, ', '.join(map(str, self.args)))

@node('target', 'expr')
class Assign:
    def eval(self, ctx):
        expr = self.expr.eval(ctx)
        # Handle assignment to slices
        if isinstance(self.target, Slice):
            hi, lo = self.target.hi.eval(ctx), self.target.lo.eval(ctx)
            width = hi - lo + 1

            assert isinstance(self.target.expr, Identifier)
            name = self.target.expr.name

            old = ctx.get(name)
            if not is_z3(old):
                old = z3.BitVec('undef', width)

            # Hack around Z3 API to get a bit vector of the expected width
            if is_z3(expr):
                expr = z3.Extract(width-1, 0, expr)
            elif width > 0:
                expr = z3.BitVecVal(expr, width)

            # Append the unassigned and assigned portions of this vector
            args = []
            if old.size()-1 >= hi+1:
                args.append(z3.Extract(old.size()-1, hi+1, old))
            if width > 0:
                args.append(expr)
            if lo-1 >= 0:
                args.append(z3.Extract(lo-1, 0, old))

            new = z3.Concat(*args) if len(args) > 1 else args[0]
            assert new.size() == old.size()

            ctx.set(name, ctx.predicate(new, old))
        # Assigning to a raw variable. Only need to deal with predication.
        else:
            assert isinstance(self.target, Identifier)
            name = self.target.name
            ctx.set(name, ctx.predicate(expr, ctx.get(name)))

        return None

    def __repr__(self):
        return '%s := %s' % (self.target, self.expr)

@node('stmts')
class Block:
    def eval(self, ctx):
        for stmt in self.stmts:
            stmt.eval(ctx)
        return None
    def __repr__(self):
        return '\n'.join(map(str, self.stmts))

@node('expr', 'if_block', 'else_block')
class If:
    def eval(self, ctx):
        expr = self.expr.eval(ctx)
        if z3.is_bv(expr) or isinstance(expr, int):
            expr = (expr != 0)
        expr = try_simplify(expr)

        # If we can statically determine this expression, only execute one branch
        bool_expr = try_bool(expr)
        if bool_expr == True:
            self.if_block.eval(ctx)
        elif bool_expr == False:
            self.else_block.eval(ctx)
        # Otherwise, execute both branches with a predicate
        else:
            with ctx.predicated(expr):
                self.if_block.eval(ctx)
            with ctx.predicated(z3.Not(expr)):
                self.else_block.eval(ctx)
        return None

    def __repr__(self):
        else_block = 'ELSE\n%s\n' % indent(self.else_block) if self.else_block.stmts else ''
        return 'IF %s\n%s\n%sFI' % (self.expr,
                indent(self.if_block), else_block)

@node('expr', 'cases')
class Case:
    def eval(self, ctx):
        for [value, stmt] in self.cases:
            expr = try_simplify(self.expr.eval(ctx) == value.eval(ctx))
            # Try to resolve the expression statically
            bool_expr = try_bool(expr)
            if bool_expr == True:
                stmt.eval(ctx)
                return None
            # Unknown expression, use predication
            elif bool_expr is None:
                with ctx.predicated(expr):
                    stmt.eval(ctx)
    def __repr__(self):
        cases = '\n'.join('%8s: %s' % (value, stmt) for [value, stmt] in self.cases)
        return 'CASE %s OF\n%s\nESAC' % (self.expr, indent(cases))

@node('var', 'lo', 'hi', 'block')
class For:
    def eval(self, ctx):
        lo, hi = self.lo.eval(ctx), self.hi.eval(ctx)
        assert lo <= hi
        assert isinstance(self.var, Identifier)
        var = self.var.name
        for x in range(lo, hi+1):
            ctx.set(var, x)
            self.block.eval(ctx)
        return None
    def __repr__(self):
        return 'FOR %s := %s to %s\n%s\nENDFOR' % (self.var,
                self.lo, self.hi, indent(self.block))

class ReturnExc(Exception):
    def __init__(self, value):
        self.value = value

@node('expr')
class Return:
    def eval(self, ctx):
        raise ReturnExc(self.expr.eval(ctx))
    def __repr__(self):
        return 'RETURN %s' % self.expr

@node('name', 'params', 'block')
class Function:
    def eval(self, ctx):
        def run(*args, **ctx_args):
            ctx = Context(**ctx_args)
            assert len(args) == len(self.params)
            for p, a in zip(self.params, args):
                ctx.set(p.name, a.eval(ctx) if isinstance(a, Node) else a)

            try:
                self.block.eval(ctx)
            except ReturnExc as e:
                return e.value

            return None

        run.__name__ = self.name
        ctx.set(self.name, run)
        return None

    def __repr__(self):
        return 'DEFINE %s(%s) {\n%s\n}' % (self.name, ', '.join(map(str, self.params)),
                indent(self.block))

# Uhh I don't think you're actually supposed to do this. Hex formatting for Z3.
class HexFormatter(z3.Formatter):
    def pp_int(self, a):
        return z3.to_format(hex(a.as_long()))
    def pp_bv(self, a):
        return z3.to_format(hex(a.as_long()))
z3.z3printer._Formatter = HexFormatter()

# Run various expressions through a solver.
def check(assertion, for_all=[]):
    solver = z3.Solver()
    if for_all:
        assertion = z3.ForAll(for_all, assertion)
    result = solver.check(assertion)
    if result != z3.sat:
        print(result)
        #print(solver.unsat_core())
        return None
    model = solver.model()
    return model
