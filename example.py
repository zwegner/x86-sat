from parse import *

parser = LazyXMLParser()
parser.parse_xml()
m256 = parser.prefixed('_mm256_')
m512 = parser.prefixed('_mm512_')

i = Var('i', 'int')
j = Var('j', 'int')
x = Var('x', '__m256i')
y = Var('y', '__m256i')
a = Var('a', '__m512i')
b = Var('b', '__m512i')
c = Var('c', '__m512i')

# XOR zeroing
check_print(m256.xor_si256(x, x) == m256.set1_epi8(0), for_all=[x])

# Ternary logic: derive the imm8 lookup table from inputs/output
check_print(m512.ternarylogic_epi32(m512.set1_epi8(0xAA),
       m512.set1_epi8(0xCC), m512.set1_epi8(0xF0), i) == m512.set1_epi8(0x57))
# -> [i = 0x1f]

# 16-bit x<<1 is the same as 16-bit x+x
check_print(m256.add_epi16(x, x) == m256.slli_epi16(x, 1), for_all=[x])

# 16-bit x<<1 is the same as 32-bit x+x after masking the low bit
check_print(m256.and_si256(m256.set1_epi16(0xFFFE), m256.add_epi32(x, x)) == m256.slli_epi16(x, 1),
        for_all=[x])

# Movemask sanity checks
check_print(m256.movemask_epi8(m256.set1_epi32(0xFF0000FF)) == i)
# -> [i = 0x99999999]
check_print(m256.movemask_epi8(m256.set1_epi32(i)) == 0x99999999)
# -> [i = 0x80000080]
check_print(m256.movemask_epi8(m256.slli_epi16(m256.set1_epi32(0x02000002), i)) == 0x99999999,
    for_all=[x])
# -> [i = 0x6]

# Find shifts of [0..n] vector to get various bitmasks
for mask in [0xFFFF0000, 0xFF00FF00, 0xF0F0F0F0, 0xCCCCCCCC, 0xAAAAAAAA]:
    check_print(m256.movemask_epi8(m256.slli_epi16(m256.setr_epi8(*range(32)), i)) == mask,
        for_all=[x])
# -> [i = 0x3] through [i = 0x7], you get the picture

# Find index vector for lookup to be the identity function
# We use &0x0F because it's too slow otherwise
check_print(x == m256.shuffle_epi8(x, m256.and_si256(y, m512.set1_epi8(0x0F))), for_all=[x])
# -> [y = 0xf0e0d0c0b0a090807060504030201000f0e0d0c0b0a09080706050403020100]

# Find an index vector with all high bits set to zero output
check_print(0 == m256.shuffle_epi8(x, m256.and_si256(y, m512.set1_epi8(0x8F))), for_all=[x])
# -> [y = 0x888084828080868c8080878288808084808981818084888e848c8a8080808082]

# Find an input index to vpermb that reverses the input
# XXX Intel has a bug in the m512.set_epi8 pseudocode, which makes this
# unsatisfiable
values = range(2, 3*64, 3)
check_print(m512.set_epi8(*values) == m512.permutexvar_epi8(b, m512.set_epi8(*reversed(values))))
# <- [b = 0x000102030405060708090a0b0c0d0e0f1011...]

# Derive permute2x128/alignr args for lane shifts
def find_shift(k):
    bottom = m256.setr_epi8(*range(32))
    top = m256.setr_epi8(*range(32, 64))
    perm = m256.permute2x128_si256(top, bottom, j)
    check_print(m256.alignr_epi8(top, perm, i) == m256.setr_epi8(*range(32-k, 64-k)))

find_shift(0)    # -> [i = 0x10, j = 0x88]
find_shift(1)    # -> [i = 0xf, j = 0x3]  
find_shift(2)    # -> [i = 0xe, j = 0x3]  
find_shift(15)   # -> [i = 0x1, j = 0x3]  
find_shift(16)   # -> [i = 0x0, j = 0x3]  
