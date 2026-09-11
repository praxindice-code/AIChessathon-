"""Offline generator for the magic-bitboard constants used by board.py.

Run with:  python tools/gen_magics.py > magics.py

The constants are plain numbers found by random search on this machine; embedding them keeps
import time on the platform deterministic and free of a search loop.
"""

import random

MASK64 = (1 << 64) - 1
ROOK_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
BISHOP_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def ray(square: int, dr: int, df: int, occupancy: int, edge: bool) -> int:
    attacks = 0
    rank, file = divmod(square, 8)
    while True:
        rank += dr
        file += df
        if not (0 <= rank < 8 and 0 <= file < 8):
            break
        target = rank * 8 + file
        # relevance mask: stop before the border square, it never blocks anything
        if edge and (not (0 < rank < 7 or dr == 0) or not (0 < file < 7 or df == 0)):
            break
        attacks |= 1 << target
        if occupancy >> target & 1:
            break
    return attacks


def attacks_for(square: int, occupancy: int, dirs: tuple[tuple[int, int], ...], edge: bool) -> int:
    return sum(ray(square, dr, df, occupancy, edge) for dr, df in dirs)


def subsets(mask: int) -> list[int]:
    bits = [b for b in range(64) if mask >> b & 1]
    out = []
    for index in range(1 << len(bits)):
        value = 0
        for position, bit in enumerate(bits):
            if index >> position & 1:
                value |= 1 << bit
        out.append(value)
    return out


def find_magic(
    square: int, dirs: tuple[tuple[int, int], ...], rng: random.Random
) -> tuple[int, int]:
    mask = attacks_for(square, 0, dirs, True)
    bits = bin(mask).count("1")
    occupancies = subsets(mask)
    references = [attacks_for(square, occupancy, dirs, False) for occupancy in occupancies]
    size = 1 << bits
    while True:
        magic = rng.getrandbits(64) & rng.getrandbits(64) & rng.getrandbits(64)
        if bin(mask * magic & 0xFF00000000000000).count("1") < 6:
            continue
        table: dict[int, int] = {}
        for occupancy, reference in zip(occupancies, references, strict=True):
            index = (occupancy * magic & MASK64) >> (64 - bits)
            if table.setdefault(index, reference) != reference:
                break
        else:
            return magic, size


def signed(value: int) -> int:
    value &= MASK64
    return value - (1 << 64) if value >> 63 else value


def main() -> None:
    rng = random.Random(20260902)
    rook = [find_magic(square, ROOK_DIRS, rng) for square in range(64)]
    bishop = [find_magic(square, BISHOP_DIRS, rng) for square in range(64)]
    print('"""Magic-bitboard multipliers, found offline by tools/gen_magics.py (seed 20260902)."""')
    print()
    print("ROOK_MAGICS = (")
    for magic, _ in rook:
        print(f"    {signed(magic)},")
    print(")")
    print()
    print("BISHOP_MAGICS = (")
    for magic, _ in bishop:
        print(f"    {signed(magic)},")
    print(")")
    total_rook = sum(size for _, size in rook)
    total_bishop = sum(size for _, size in bishop)
    print()
    print(f"# table entries: rook {total_rook}, bishop {total_bishop}")


if __name__ == "__main__":
    main()
