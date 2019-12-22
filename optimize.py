import itertools
import subprocess

from parse import *

TERM_WIDTH = None
def print_status(line):
    global TERM_WIDTH
    if TERM_WIDTH is None:
        # Get the terminal width (*n*x platforms only) for status printing
        try:
            TERM_WIDTH = int(subprocess.check_output(['stty', 'size']).split()[0])
        except Exception:
            TERM_WIDTH = 80
    line = (line + ' ' * TERM_WIDTH)[:TERM_WIDTH]
    print(line, end='\r')

# Exception to break out of recursive search when done
class DoneExc(Exception):
    pass

def optimize(inputs, output, free_vars=[]):
    # HACK: keep refs to all expressions evaluated so id(expr) is stable
    checked = []
    seen = set()
    nodes = 0

    # Leaf node: check if any expressions are equivalent to the output
    def check_exprs(exprs):
        nonlocal nodes
        for expr in exprs:
            # Skip already-tried expressions
            if id(expr) in seen:
                continue
            seen.add(id(expr))
            checked.append(expr)

            # Check this expression
            nodes += 1
            print_status('[%5s] %s' % (nodes, expr))
            (result, model) = check(expr == output, for_all=inputs)
            if model is not None:
                print('success!', expr)
                print(model)
                raise DoneExc()
            elif result != z3.unsat:
                print('timeout:')

    def search(exprs, depth):
        # Loop over all intrinsics
        for intr in intrinsics.values():
            # Find which expressions can be passed to each parameter
            valid_args = []
            for param in intr.params:
                valid_args.append([expr for expr in exprs
                    if expr._size == param._size])

            # Search through every argument combination
            for args in itertools.product(*valid_args):
                child_exprs = [intr(*args)] + exprs
                if depth <= 1:
                    check_exprs(child_exprs)
                else:
                    search(child_exprs, depth - 1)

    # Search deeper and deeper until we find something
    exprs = [*inputs, *free_vars]
    try:
        for depth in range(1, 10):
            print('depth', depth)
            search(exprs, depth)
            print_status('')
    except DoneExc:
        print('done, nodes=%s' % nodes)

# Parse the intrinsics used in search and put them in global scope
metadata = parse_meta('data.xml')
regex = '|'.join([
    r'_mm256_and_si256',
    r'_mm256_mullo_epi32',
    r'_mm256_srli_epi32',
])
intrinsics = parse_whitelist(metadata, regex=regex)
globals().update(intrinsics)
# Non-searched regexes
regex = '|'.join([
    r'_mm256_set1_epi(8|16|32)',
    r'_mm256_popcnt_epi32',
])
globals().update(parse_whitelist(metadata, regex=regex))

# Test problem: find positional popcount of bit 0 per byte, summed over dwords
# Target solution: (x & 0x01010101) * 0x01010101 >> 24
x = Var('x', '__m256i')
output = _mm256_popcnt_epi32(_mm256_and_si256(x, _mm256_set1_epi8(0x01)))

free_vars = [
    # HACK: reduce search space by making all bytes the same
    _mm256_set1_epi8(Var('i', 'int')),
    Var('j', 'int'),
]

# HACK: set a timeout for proving/disproving each expression. It's easy for
# Z3 to get stuck on some obviously-wrong expression if it's complicated enough
SOLVER.set('timeout', 10000)

optimize([x], output, free_vars=free_vars)
