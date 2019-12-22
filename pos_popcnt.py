from parse import *

# Parse the intrinsics used and put them in global scope
regex = '|'.join([
    r'_mm512_set(1)?_epi(8|64)',
    r'_mm512_gf2p8affine_epi64_epi8',
    r'_mm512_permutexvar_epi8',
    r'_mm512_(and|srli|popcnt|reduce_add)_epi64',
])
globals().update(parse_whitelist('data.xml', regex=regex))

def pos_popcnt_fast(in_bytes):
    # Rotate bits within each qword, so that bit 0 of every byte is contiguous in byte 0, etc
    # Credit to Geoff Langdale for this trick:
    # https://branchfree.org/2019/05/29/why-ice-lake-is-important-a-bit-bashers-perspective
    x = _mm512_set1_epi64(0x8040201008040201)
    rotated = _mm512_gf2p8affine_epi64_epi8(x, in_bytes, 0)

    # Gather bytes 0,8,16, etc into qword 0, 1,9,17 etc into qword 1, etc etc
    # reversed() since _mm512_set_epi8 is big-endian
    indices = _mm512_set_epi8(*reversed([i + j for i in range(8) for j in range(0, 64, 8)]))
    gathered = _mm512_permutexvar_epi8(indices, rotated)

    return _mm512_popcnt_epi64(gathered)

def pos_popcnt_slow(in_bytes):
    popcnts = []
    for i in range(8):
        # Mask out bit i in each byte
        shifted = _mm512_srli_epi64(in_bytes, i)
        mask = _mm512_and_epi64(shifted, _mm512_set1_epi8(0x01))
        # Count the bits in each qword
        popcnt = _mm512_popcnt_epi64(mask)
        # Sum across all qwords, add to the list
        popcnts.append(_mm512_reduce_add_epi64(popcnt))
    # Return result vector (reversed for big-endian)
    return _mm512_set_epi64(*reversed(popcnts))

a = Var('a', '__m512i')

check_print(pos_popcnt_fast(a) == pos_popcnt_slow(a), for_all=[a])
