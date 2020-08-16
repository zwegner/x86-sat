import contextlib
import inspect
import re
import sys

import z3

from . import intr_builtins

# Context handles the current symbol values during execution, and predication.
# Predication is required for handling branches on unknown data, which get
# transformed into a Z3 If() over the results of both branches.
# The parent context is passed in as well, for accessing symbols in parent
# scopes. I guess this is technically "dynamic scoping", but I doubt this makes
# a difference vs. actual lexical scoping
class Context:
    def __init__(self, pred=None, parent=None):
        self.symbols = {}
        self.parent = parent
        self.pred = pred
    def set(self, name, value):
        self.symbols[name] = value
    def get(self, name):
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.get(name)
        return getattr(intr_builtins, name, None)

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

# Kinda hacky: evaluate a Node if it's a Node. This is basically just an
# alternative to making a wrapper node type for literals
def try_eval(ctx, e):
    return e.eval(ctx) if isinstance(e, Node) else e

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
    if z3.is_bv(b) or isinstance(b, int):
        b = (b != 0)
    b = try_simplify(b)
    # HACK: z3 python interface has bug/weird behavior where (x == y) is
    # always False for unknown x and y, so use is_true and is_false instead
    if z3.is_true(b):
        return (True, None)
    if z3.is_false(b):
        return (False, None)
    return (None, b)

def zero_ext(value, width):
    if not is_z3(value):
        return value
    # XXX is this safe?
    if not z3.is_bv(value):
        value = z3.Int2BV(value, width)
    assert value.size() > 0
    diff = try_simplify(width - value.size())
    if diff > 0:
        return z3.ZeroExt(diff, value)
    return value

# Make sure two operands are the same width by zero-extending the smaller one.
def match_width(lhs, rhs):
    if is_z3(lhs) or is_z3(rhs):
        widths = [v.size() for v in [lhs, rhs] if z3.is_bv(v)]
        if widths:
            width = max(widths)
            return [zero_ext(lhs, width), zero_ext(rhs, width), widths]
    return [lhs, rhs, None]

# The "add" argument adds more bits, and is needed at least by left shift.
# Intel's code uses stuff like (bit << 2), which needs to be 3 bits, not 1.
# "double" doubles the width. This is needed for multiplications, which
# could silently overflow before.
def match_width_fn(lhs, rhs, fn, add=0, double=False, widths=None):
    if is_z3(lhs) or is_z3(rhs):
        if widths is None:
            widths = [v.size() for v in [lhs, rhs] if z3.is_bv(v)]
        if widths:
            width = max(widths) + add
            if double:
                width *= 2
            return fn(zero_ext(lhs, width), zero_ext(rhs, width))
    return fn(lhs, rhs)

# For pretty printing
def indent(s):
    return '    ' + str(s).replace('\n', '\n    ')

# Width of supported C types in bits
WIDTH_TYPES = {
      8: ['int8_t',  'uint8_t', 'char', '__mmask8'],
     16: ['int16_t', 'uint16_t', 'short', '__mmask16'],
     32: ['int32_t', 'uint32_t', 'int', 'const int', 'unsigned int', '__mmask32'],
     64: ['int64_t', 'uint64_t', 'long long', '__int64', 'unsigned __int64', '__mmask64', '__m64'],
    128: ['__m128i'],
    256: ['__m256', '__m256d', '__m256i'],
    512: ['__m512i']
}
TYPE_WIDTH = {t: size for size, ts in WIDTH_TYPES.items() for t in ts}

# Deep structural equality check for Node types. Node.__eq__ is overloaded to
# create a new BinaryOp expression, so this needs to be a separate function
def equal(a, b):
    if type(a) != type(b):
        return False
    if not isinstance(a, Node):
        return a == b
    return all(equal(getattr(a, param), getattr(b, param))
            for param in type(a).params)

# AST types. These handle all evaluation

class Node:
    # This generic __init__ uses the params/kwparams filled in by the
    # @node() decorator
    def __init__(self, *args, info=None, **kwargs):
        params, kwparams = type(self).params, type(self).kwparams
        assert len(args) == len(params)
        for p, a in zip(params, args):
            setattr(self, p, a)

        # Update attributes from default parameters, then actual kwargs
        for k, v in kwparams.items():
            setattr(self, k, v)
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.info = info
        if self.info is None:
            for arg in args:
                if isinstance(arg, Node) and arg.info is not None:
                    self.info = arg.info
                    break

        if hasattr(self, 'setup'):
            self.setup()

    # Overload ops to create new expressions
    def __eq__(self, other):      return BinaryOp('==',  self, other)
    def __ne__(self, other):      return BinaryOp('!=',  self, other)
    def __add__(self, other):     return BinaryOp('+',   self, other)
    def __radd__(self, other):    return BinaryOp('+',   self, other)
    def __sub__(self, other):     return BinaryOp('-',   self, other)
    def __rsub__(self, other):    return BinaryOp('-',   other, self)
    def __lshift__(self, other):  return BinaryOp('<<',  self, other)
    def __rshift__(self, other):  return BinaryOp('>>',  self, other)
    def __and__(self, other):     return BinaryOp('AND', self, other)
    def __or__(self, other):      return BinaryOp('OR',  self, other)
    def __xor__(self, other):     return BinaryOp('XOR', self, other)

    def __invert__(self):         return UnaryOp('NOT',  self)
    def __neg__(self):            return UnaryOp('-',  self)

    def __getitem__(self, item):
        if isinstance(item, slice):
            assert item.start is not None
            assert item.stop is not None
            assert item.step is None
            return Slice(self, Integer(item.start), Integer(item.stop))
        return Slice(self, None, Integer(item))

# Decorator for easily making Node subclasses with given parameters
def node(*params, **kwparams):
    def decorate(cls):
        # Redefine class to have Node as parent. Cool hack bro.
        cls = type(cls.__name__, (Node,), dict(cls.__dict__))
        cls.params = params
        cls.kwparams = kwparams
        return cls
    return decorate

def get_type_width(t):
    return TYPE_WIDTH[t]

def cast(t, val):
    return z3.BitVecVal(val, TYPE_WIDTH[t])

def serialize(value, t):
    width = TYPE_WIDTH[t]
    [n, m] = divmod(value.size(), width)
    assert m == 0
    return [z3.simplify(z3.Extract((i + 1) * width - 1, i * width, value)) for i in range(n)]

# Generic free variable, evaluates to a Z3 bitvector with the right number of
# bits for the corresponding C type
@node('name', 'type')
class Var:
    def setup(self):
        self._size = get_type_width(self.type)
    def eval(self, ctx):
        return z3.BitVec(self.name, self._size)
    def __repr__(self):
        return self.name

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
        elif self.op == '-':
            return -self.rhs.eval(ctx)
        assert False
    def __repr__(self):
        return '(%s %s)' % (self.op, self.rhs)

@node('op', 'lhs', 'rhs')
class BinaryOp:
    def eval(self, ctx):
        [lhs, rhs, widths] = match_width(try_eval(ctx, self.lhs),
                try_eval(ctx, self.rhs))
        if self.op == '+':
            return lhs + rhs
        elif self.op == '-':
            return lhs - rhs
        elif self.op == '*':
            return match_width_fn(lhs, rhs, lambda l, r: l * r,
                    double=True, widths=widths)
        elif self.op == 'AND':
            return lhs & rhs
        elif self.op == 'OR':
            return lhs | rhs
        elif self.op == 'XOR':
            return lhs ^ rhs
        elif self.op == '<<':
            # Add more bits to the left if we know the rhs
            add = rhs if isinstance(rhs, (int, Integer)) else 0
            return match_width_fn(lhs, rhs, lambda l, r: l << r,
                    add=add, widths=widths)
        elif self.op == '>>':
            return lhs >> rhs
        elif self.op == '<':
            return lhs < rhs
        elif self.op == '>':
            return lhs > rhs
        elif self.op == '==':
            return lhs == rhs
        elif self.op == '!=':
            return lhs != rhs
        assert False, 'unknown binop %s' % self.op
    def __repr__(self):
        return '(%s %s %s)' % (self.lhs, self.op, self.rhs)

# Attribute access. Intel uses this for treating variables like pseudo-unions
# containing arrays of variously sized elements, like x.byte[y]
SCALE_SIZE = {
    'bit': 1,
    'byte': 8,
    'word': 16,
    'dword': 32,
    'qword': 64
}
@node('expr', 'attr')
class Attr:
    def get_scale(self):
        return SCALE_SIZE[self.attr]
    def __repr__(self):
        return '%s.%s' % (self.expr, self.attr)

# Reduce an identifier with possible nested attribute accesses/array indices
# (example: dst.qword[j].byte[i]) into a few pieces of information, suitable
# for both reading from and writing to this value. This returns a tuple of:
# * name, which is the identifier string containing the underlying BitVec
#   ('dst' in the example above), or an expression, in which case this access
#   is read-only
# * lo/hi, which are the lowest/highest bits being selected
# * scale, which is the size of each element (bit==1, byte==8, etc)
# * width, which is how many total bits are being selected
def get_range(self, ctx):
    if isinstance(self, Identifier):
        return (self.name, 0, None, 1, 1)
    elif isinstance(self, Attr):
        # Scale unused
        [name, lo, hi, _, width] = get_range(self.expr, ctx)
        assert hi is None
        scale = self.get_scale()
        return (name, lo, hi, scale, scale)
    elif isinstance(self, Slice):
        [name, base_lo, base_hi, scale, width] = get_range(self.expr, ctx)
        assert base_hi is None

        lo = self.lo.eval(ctx)
        if self.hi is None:
            hi = None
        else:
            hi = base_lo + scale * self.hi.eval(ctx)
            width = hi - lo + 1
        return (name, base_lo + lo * scale, hi, scale, width)
    # Normal expressions: we return a node as a name, which can be used
    # for read-only accesses. We double check during assignment that
    # the name is a string, as returned from the Identifier case above
    else:
        return (self, 0, None, 1, 1)
    assert False

@node('expr', 'hi', 'lo')
class Slice:
    def eval(self, ctx):
        [name, lo, hi, scale, width] = get_range(self, ctx)
        expr = ctx.get(name) if isinstance(name, str) else name.eval(ctx)

        if is_z3(expr):
            expr = match_width_fn(expr, lo, lambda l, r: l >> r)

            # Big hack! Simplify (x+y)-x -> y to get the width when we don't
            # know x. This is pretty common, e.g. a[index*8+7:index*8]
            if (isinstance(self.hi, BinaryOp) and self.hi.op == '+' and
                    equal(self.hi.lhs, self.lo)):
                width1 = try_simplify(self.hi.rhs.eval(ctx))
                return z3.Extract(width1, 0, expr)

            return z3.Extract(width - 1, 0, expr)

        # Slice integers with normal bit ops
        assert width > 0
        mask = ((1 << width) - 1)
        return (expr >> lo) & mask

    def __repr__(self):
        if self.hi is None:
            return '%s[%s]' % (self.expr, self.lo)
        return '%s[%s:%s]' % (self.expr, self.hi, self.lo)

@node('target', 'expr')
class Assign:
    def eval(self, ctx):
        expr = self.expr.eval(ctx)
        # Handle assignment to slices
        if isinstance(self.target, Slice):
            [name, lo, hi, scale, width] = get_range(self.target, ctx)

            assert isinstance(name, str), 'slice assignment to non-identifier: %s' % name

            old = ctx.get(name)
            if not is_z3(old):
                old = z3.BitVec('undef', width)

            # Hack around Z3 API to get a bit vector of the expected width
            if is_z3(expr):
                expr = z3.Extract(width - 1, 0, expr)
            elif width > 0:
                expr = z3.BitVecVal(expr, width)

            if hi is None:
                hi = lo + scale - 1
            # Append the unassigned and assigned portions of this vector
            args = []
            if old.size() - 1 >= hi + 1:
                args.append(z3.Extract(old.size() - 1, hi + 1, old))
            if width > 0:
                args.append(expr)
            if lo - 1 >= 0:
                args.append(z3.Extract(lo - 1, 0, old))

            new = z3.Concat(*args) if len(args) > 1 else args[0]

            # XXX we can't always rely on this, think of a better way to check
            #assert new.size() == old.size()

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
        # If we can statically resolve this condition, only execute one branch
        (bool_expr, expr) = try_bool(self.expr.eval(ctx))
        if bool_expr is True:
            return self.if_block.eval(ctx)
        elif bool_expr is False:
            return self.else_block.eval(ctx)
        # Otherwise, execute both branches with a predicate
        else:
            with ctx.predicated(expr):
                if_expr = self.if_block.eval(ctx)
            with ctx.predicated(z3.Not(expr)):
                else_expr = self.else_block.eval(ctx)
            if if_expr or else_expr:
                return if_expr | else_expr
            return None

    def __repr__(self):
        else_block = ('ELSE\n%s\n' % indent(self.else_block)
                if isinstance(self.else_block, Block) and self.else_block.stmts else '')
        return 'IF %s\n%s\n%sFI' % (self.expr,
                indent(self.if_block), else_block)

@node('expr', 'cases')
class Case:
    def eval(self, ctx):
        for [value, stmt] in self.cases:
            # Try to resolve the expression statically
            (bool_expr, expr) = try_bool(self.expr.eval(ctx) == value.eval(ctx))
            if bool_expr is True:
                stmt.eval(ctx)
                return None
            # Unknown expression, use predication
            elif bool_expr is None:
                with ctx.predicated(expr):
                    stmt.eval(ctx)
    def __repr__(self):
        cases = '\n'.join('%8s: %s' % (value, stmt)
                for [value, stmt] in self.cases)
        return 'CASE %s OF\n%s\nESAC' % (self.expr, indent(cases))

@node('var', 'lo', 'hi', 'block', step=1)
class For:
    def eval(self, ctx):
        lo, hi = self.lo.eval(ctx), self.hi.eval(ctx)
        # This isn't always true: _mm_clmulepi64_si128 at least has '1 to 0'
        #assert lo <= hi
        assert isinstance(self.var, Identifier)
        var = self.var.name
        # Handle ranges in either direction, with inclusive start/stop
        for x in range(lo, hi + self.step, self.step):
            ctx.set(var, x)
            self.block.eval(ctx)
        return None
    def __repr__(self):
        return 'FOR %s := %s to %s\n%s\nENDFOR' % (self.var,
                self.lo, self.hi, indent(self.block))

@node('expr', 'block')
class While:
    def eval(self, ctx):
        while True:
            # We can only break if we statically know the condition to be false.
            # While loops in hardware specs are kinda weird.
            (bool_expr, expr) = try_bool(self.expr.eval(ctx))
            if bool_expr is False:
                break
            # Loop condition definitely true: just execute the body
            elif bool_expr is True:
                self.block.eval(ctx)
            # Otherwise, execute the loop body with a predicate
            else:
                with ctx.predicated(expr):
                    self.block.eval(ctx)
        return None
    def __repr__(self):
        return 'DO WHILE %s\n%s\nOD' % (self.expr, indent(self.block))

class ReturnExc(Exception):
    def __init__(self, value):
        self.value = value

@node('expr')
class Return:
    def eval(self, ctx):
        raise ReturnExc(self.expr.eval(ctx))
    def __repr__(self):
        return 'RETURN %s' % self.expr

@node('fn', 'args')
class Call:
    def eval(self, ctx):
        fn = self.fn.eval(ctx)
        if isinstance(fn, Function):
            fn = fn.run
        args = [try_eval(ctx, a) for a in self.args]
        return fn(*args, pred=ctx.pred, parent=ctx)
    def __repr__(self):
        return '%s(%s)' % (self.fn, ', '.join(map(str, self.args)))

@node('name', 'params', 'block', return_type=None)
class Function:
    def run(self, *args, **ctx_args):
        ctx = Context(**ctx_args)
        assert len(args) == len(self.params)
        for p, a in zip(self.params, args):
            ctx.set(p.name, a)

        try:
            self.block.eval(ctx)
        except ReturnExc as e:
            return e.value

        return None

    def eval(self, ctx):
        ctx.set(self.name, self)
        return self

    def __call__(self, *args):
        # This will cause an extra eval() of the actual function each time it's
        # called, which calls ctx.set(). This should pretty much not matter,
        # but it seems weird
        return Call(self, args, _size=getattr(self, '_size', None))

    def __repr__(self):
        if self.block is None:
            return self.name
        return 'DEFINE %s(%s) {\n%s\n}' % (self.name,
                ', '.join(map(str, self.params)), indent(self.block))

# Uhh I don't think you're actually supposed to do this. Hex formatting for Z3.
class HexFormatter(z3.Formatter):
    def pp_bv(self, a):
        return z3.to_format('0x%0*x' % (a.size() // 4, a.as_long()))
z3.z3printer._Formatter = HexFormatter()

@contextlib.contextmanager
def munge_exceptions():
    try:
        yield
    except Exception as e:
        traceback = inspect.trace()
        # XXX this is pretty hacky!
        print('Pseudocode traceback:')
        last_index = 0
        for i, (frame, *_) in enumerate(traceback):
            self = frame.f_locals.get('self', None)
            if isinstance(self, Node) and not isinstance(self, Block):
                last_index = i
                node_type = type(self).__name__
                if self.info:
                    print('File "%s", line %s, in %s' % (self.info.filename,
                        self.info.lineno, node_type))
                else:
                    print('<unknown source>, in %s' % node_type)
                # Print the first line of the "source"
                source = str(self).strip()
                source = re.sub(r'^([^\n]*(?=\n)|.{70}).*$', r'\1...', source,
                        flags=re.DOTALL)
                print('    ', source)
        print('----------')
        print('Python traceback (below pseudocode traceback):')
        for _, filename, line, fn, ctx, index in traceback[last_index:]:
            print('  File "%s", line %s, in %s' % (filename, line, fn))
            print('    ', ctx[index].strip())
        print(e)
        sys.exit(1)

# Run various expressions through a solver.
SOLVER = z3.Solver()
def check(*assertions, for_all=[], return_type='str'):
    ctx = Context()
    with munge_exceptions():
        assertions = [assertion.eval(ctx) for assertion in assertions]
    if for_all:
        for_all = [f.eval(ctx) for f in for_all]
        assertions = [z3.ForAll(for_all, v) for v in assertions]
    SOLVER.reset()
    result = SOLVER.check(*assertions)
    if result != z3.sat:
        return (result, None)
    # The expression is satisfiable. Create a dictionary from the resulting
    # model, since the model returned by Z3 uses Z3 variables as keys, which
    # are only created on-demand in eval() above, thus not very useful outside
    # of this function
    model = SOLVER.model()
    if return_type == 'str':
        model = {v.name(): '%#0*x' % ((model[v].size() + 3) // 4 + 2, model[v].as_long()) for v in model}
    elif return_type == 'int':
        model = {v.name(): model[v].as_long() for v in model}
    elif return_type == 'var':
        model = {v.name(): model[v] for v in model}
    elif return_type.startswith('serial'):
        etype = return_type[7:]
        model = {v.name(): serialize(model[v], etype) for v in model}
    else:
        assert False, 'return type must be one of "str", "int", "var", or "serial:<type>"'
    return (result, model)

def check_print(*assertions, for_all=[], return_type='str'):
    (result, model) = check(*assertions, for_all=for_all, return_type=return_type)
    if model:
        width = max(len(k) for k in model)
        for k, v in model.items():
            print('%*s: %s' % (width, k, v))
        print()
    return model
