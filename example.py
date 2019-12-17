from parse import *

whitelist = '''\
#_mm256_load_si256
#_mm256_lddqu_si256
_mm256_set1_epi8
_mm256_set1_epi16
_mm256_set1_epi32
_mm256_and_si256
_mm256_andnot_si256
_mm256_or_si256
_mm256_xor_si256
_mm256_add_epi16
_mm256_add_epi32
_mm256_sub_epi16
_mm256_sub_epi32
_mm256_slli_epi16
_mm256_srli_epi16
#_mm256_testz_si256
_mm256_movemask_epi8
_mm256_shuffle_epi8
_mm256_setr_epi8
_mm256_permute2x128_si256
_mm256_alignr_epi8
_mm512_set1_epi8
_mm512_ternarylogic_epi32
'''.splitlines()

table = parse_instrinsics(whitelist=whitelist)

# Just stick em all in globals! Who cares?!
globals().update(table)

a = var('a', '__m512i')
b = var('b', '__m512i')
c = var('c', '__m512i')
x = var('x', '__m256i')
y = var('y', 'int')
z = var('z', 'int')

# XOR zeroing
check(_mm256_xor_si256(x, x) == _mm256_set1_epi8(0), for_all=[x])

# Ternary logic: derive the imm8 lookup table from inputs/output
check(_mm512_ternarylogic_epi32(_mm512_set1_epi8(0xAA),
       _mm512_set1_epi8(0xCC), _mm512_set1_epi8(0xF0), y) == _mm512_set1_epi8(0x57))
# -> [y = 0x1f]

# 16-bit x<<1 is the same as 16-bit x+x
check(_mm256_add_epi16(x, x) == _mm256_slli_epi16(x, 1), for_all=[x])

# 16-bit x<<1 is the same as 32-bit x+x after masking the low bit
check(_mm256_and_si256(_mm256_set1_epi16(0xFFFE), _mm256_add_epi32(x, x)) == _mm256_slli_epi16(x, 1),
        for_all=[x])
# -> [y = 0x6]

# Movemask sanity checks
check(_mm256_movemask_epi8(_mm256_set1_epi32(0xFF0000FF)) == y)
# -> [y = 0x99999999]
check(_mm256_movemask_epi8(_mm256_set1_epi32(y)) == 0x99999999)
# -> [y = 0x80000080]
check(_mm256_movemask_epi8(_mm256_slli_epi16(_mm256_set1_epi32(0x02000002), y)) == 0x99999999,
    for_all=[x])
# -> [y = 0x6]

# Find shifts of [0..n] vector to get various bitmasks
for mask in [0xFFFF0000, 0xFF00FF00, 0xF0F0F0F0, 0xCCCCCCCC, 0xAAAAAAAA]:
    check(_mm256_movemask_epi8(_mm256_slli_epi16(_mm256_setr_epi8(*range(32)), y)) == mask,
        for_all=[x])
# -> [y = 0x3] through [y = 0x7], you get the picture

# Find index vector for lookup to be the identity function
# We use &0x0F because it's too slow otherwise
check(x == _mm256_shuffle_epi8(x, _mm256_and_si256(b, _mm512_set1_epi8(0x0F))), for_all=[x])
# -> [b = 0xf0e0d0c0b0a090807060504030201000f0e0d0c0b0a09080706050403020100]

# Find an index vector with all high bits set to zero output
check(0 == _mm256_shuffle_epi8(x, _mm256_and_si256(b, _mm512_set1_epi8(0x8F))), for_all=[x])
# -> [b = 0x888084828080868c8080878288808084808981818084888e848c8a8080808082]

# Derive permute2x128/alignr args for lane shifts
def find_shift(i):
    bottom = _mm256_setr_epi8(*range(32))
    top = _mm256_setr_epi8(*range(32, 64))
    perm = _mm256_permute2x128_si256(top, bottom, y)
    check(_mm256_alignr_epi8(top, perm, z) == _mm256_setr_epi8(*range(32-i, 64-i)))

find_shift(0)    # -> [z = 0x10, y = 0x88]
find_shift(1)    # -> [z = 0xf, y = 0x3]  
find_shift(2)    # -> [z = 0xe, y = 0x3]  
find_shift(15)   # -> [z = 0x1, y = 0x3]  
find_shift(16)   # -> [z = 0x0, y = 0x3]  
