from parse import *

# Parse only a narrow subset of intrinsics used here, for speed
regex = '|'.join([
    r'_mm256_set1_epi(8|16|32)',
    r'_mm256_(setr|movemask|shuffle)_epi8',
    r'_mm256_(and|or|xor)_si256',
    r'_mm256_add_epi(16|32)',
    r'_mm256_slli_epi16',
    r'_mm256_(permute2x128|alignr_epi8)',
    r'_mm512_set(1)?_epi8',
    r'_mm512_ternarylogic_epi32',
    r'_mm512_permutexvar_epi8'
])
intrinsics = parse_whitelist('data.xml', regex=regex)
# Stick em all in global scope
globals().update(intrinsics)

i = Var('i', 'int')
j = Var('j', 'int')
x = Var('x', '__m256i')
y = Var('y', '__m256i')
a = Var('a', '__m512i')
b = Var('b', '__m512i')
c = Var('c', '__m512i')

def print_status(line):
    status = (line[:77] + '...' + ' ' * 80)[:80]
    print(status, end='\r')

PASSES = N_TESTS = 0

def test(expr, model=None, model_fn=None, **kwargs):
    global PASSES, N_TESTS
    N_TESTS += 1
    print_status('Test %2s: %s' % (N_TESTS, expr))
    (result, m) = check(expr, **kwargs)
    if model_fn:
        m = model_fn(m)
    if result == z3.sat and m == model:
        PASSES += 1
    else:
        print_status('')
        print('FAIL!\n    expr: %s\n   model: %s' % (expr, m))

# XOR zeroing
test(_mm256_xor_si256(x, x) == _mm256_set1_epi8(0), for_all=[x],
        model={})

# Ternary logic: derive the imm8 lookup table from inputs/output
test(_mm512_ternarylogic_epi32(_mm512_set1_epi8(0xAA),
       _mm512_set1_epi8(0xCC), _mm512_set1_epi8(0xF0), i) == _mm512_set1_epi8(0x57),
       model={'i': 0x0000001f})

# 16-bit x<<1 is the same as 16-bit x+x
test(_mm256_add_epi16(x, x) == _mm256_slli_epi16(x, 1), for_all=[x],
        model={})

# 16-bit x<<1 is the same as 32-bit x+x after masking the low bit
test(_mm256_and_si256(_mm256_set1_epi16(0xFFFE), _mm256_add_epi32(x, x)) == _mm256_slli_epi16(x, 1), for_all=[x],
        model={})

# Movemask sanity checks
test(_mm256_movemask_epi8(_mm256_set1_epi32(0xFF0000FF)) == i,
        model={'i': 0x99999999})
test(_mm256_movemask_epi8(_mm256_set1_epi32(i)) == 0x99999999,
        model={'i': 0x80000080})
test(_mm256_movemask_epi8(_mm256_slli_epi16(_mm256_set1_epi32(0x02000002), i)) == 0x99999999, for_all=[x],
    model={'i': 0x00000006})

# Find shifts of [0..n] vector to get various bitmasks
for shift, mask in enumerate([0xFFFF0000, 0xFF00FF00, 0xF0F0F0F0, 0xCCCCCCCC, 0xAAAAAAAA]):
    test(_mm256_movemask_epi8(_mm256_slli_epi16(_mm256_setr_epi8(*range(32)), i)) == mask,
            model={'i': shift + 3})

# Find index vector for lookup to be the identity function
# We use &0x0F because it's too slow otherwise
test(x == _mm256_shuffle_epi8(x, _mm256_and_si256(y, _mm256_set1_epi8(0x0F))), for_all=[x],
        model={'y': 0x0f0e0d0c0b0a090807060504030201000f0e0d0c0b0a09080706050403020100})

# Find an index vector with all high bits set to zero output
# All the bits except the high bit of each byte don't matter and are essentially random,
# so only test the high bits
high_bits = 0x8080808080808080808080808080808080808080808080808080808080808080
test(0 == _mm256_shuffle_epi8(x, _mm256_and_si256(y, _mm256_set1_epi8(0x8F))), for_all=[x],
        model_fn=lambda m: m['y'] & high_bits, model=high_bits)

# Find an input index to vpermb that reverses the input
# XXX Intel has a bug in the _mm512_set_epi8 pseudocode, which makes this
# unsatisfiable
values = range(2, 3*64, 3)
test(_mm512_set_epi8(*values) == _mm512_permutexvar_epi8(b, _mm512_set_epi8(*reversed(values))),
        model={'b': 0x000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f})

# Derive permute2x128/alignr args for lane shifts
def find_shift(k):
    bottom = _mm256_setr_epi8(*range(32))
    top = _mm256_setr_epi8(*range(32, 64))
    perm = _mm256_permute2x128_si256(top, bottom, j)
    return _mm256_alignr_epi8(top, perm, i) == _mm256_setr_epi8(*range(32-k, 64-k))

test(find_shift(0),  model={'i': 0x00000010, 'j': 0x00000088})
test(find_shift(1),  model={'i': 0x0000000f, 'j': 0x00000003})
test(find_shift(2),  model={'i': 0x0000000e, 'j': 0x00000003})
test(find_shift(15), model={'i': 0x00000001, 'j': 0x00000003})
test(find_shift(16), model={'i': 0x00000000, 'j': 0x00000003})

print_status('')
print('Passed %s/%s tests.' % (PASSES, N_TESTS))
