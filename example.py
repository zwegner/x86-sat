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

# XOR zeroing
check_print(_mm256_xor_si256(x, x) == _mm256_set1_epi8(0), for_all=[x])

# Ternary logic: derive the imm8 lookup table from inputs/output
check_print(_mm512_ternarylogic_epi32(_mm512_set1_epi8(0xAA),
       _mm512_set1_epi8(0xCC), _mm512_set1_epi8(0xF0), i) == _mm512_set1_epi8(0x57))
# -> [i = 0x1f]

# 16-bit x<<1 is the same as 16-bit x+x
check_print(_mm256_add_epi16(x, x) == _mm256_slli_epi16(x, 1), for_all=[x])

# 16-bit x<<1 is the same as 32-bit x+x after masking the low bit
check_print(_mm256_and_si256(_mm256_set1_epi16(0xFFFE), _mm256_add_epi32(x, x)) == _mm256_slli_epi16(x, 1),
        for_all=[x])

# Movemask sanity checks
check_print(_mm256_movemask_epi8(_mm256_set1_epi32(0xFF0000FF)) == i)
# -> [i = 0x99999999]
check_print(_mm256_movemask_epi8(_mm256_set1_epi32(i)) == 0x99999999)
# -> [i = 0x80000080]
check_print(_mm256_movemask_epi8(_mm256_slli_epi16(_mm256_set1_epi32(0x02000002), i)) == 0x99999999,
    for_all=[x])
# -> [i = 0x6]

# Find shifts of [0..n] vector to get various bitmasks
for mask in [0xFFFF0000, 0xFF00FF00, 0xF0F0F0F0, 0xCCCCCCCC, 0xAAAAAAAA]:
    check_print(_mm256_movemask_epi8(_mm256_slli_epi16(_mm256_setr_epi8(*range(32)), i)) == mask,
        for_all=[x])
# -> [i = 0x3] through [i = 0x7], you get the picture

# Find index vector for lookup to be the identity function
# We use &0x0F because it's too slow otherwise
check_print(x == _mm256_shuffle_epi8(x, _mm256_and_si256(y, _mm256_set1_epi8(0x0F))), for_all=[x])
# -> [y = 0xf0e0d0c0b0a090807060504030201000f0e0d0c0b0a09080706050403020100]

# Find an index vector with all high bits set to zero output
check_print(0 == _mm256_shuffle_epi8(x, _mm256_and_si256(y, _mm256_set1_epi8(0x8F))), for_all=[x])
# -> [y = 0x888084828080868c8080878288808084808981818084888e848c8a8080808082]

# Find an input index to vpermb that reverses the input
# XXX Intel has a bug in the _mm512_set_epi8 pseudocode, which makes this
# unsatisfiable
values = range(2, 3*64, 3)
check_print(_mm512_set_epi8(*values) == _mm512_permutexvar_epi8(b, _mm512_set_epi8(*reversed(values))))
# <- [b = 0x000102030405060708090a0b0c0d0e0f1011...]

# Derive permute2x128/alignr args for lane shifts
def find_shift(k):
    bottom = _mm256_setr_epi8(*range(32))
    top = _mm256_setr_epi8(*range(32, 64))
    perm = _mm256_permute2x128_si256(top, bottom, j)
    check_print(_mm256_alignr_epi8(top, perm, i) == _mm256_setr_epi8(*range(32-k, 64-k)))

find_shift(0)    # -> [i = 0x10, j = 0x88]
find_shift(1)    # -> [i = 0xf, j = 0x3]  
find_shift(2)    # -> [i = 0xe, j = 0x3]  
find_shift(15)   # -> [i = 0x1, j = 0x3]  
find_shift(16)   # -> [i = 0x0, j = 0x3]  
