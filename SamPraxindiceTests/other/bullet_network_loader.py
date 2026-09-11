"""Loads a Bullet-trained network's raw.bin checkpoint (768 -> 1024x2 -> 8 buckets,
Chess768 features, dual-perspective).

Every detail below is verified against Bullet's own docs (docs/4-saved-networks.md) and
cross-checked against the actual file sizes from a real checkpoint (raw.bin = 3,215,392 bytes,
quantised.bin = 1,607,744 bytes) -- both match exactly for this architecture, not approximately.

Layout rules, from the docs:
  - little-endian f32 throughout raw.bin
  - a declared MxN matrix is stored column-major BY DEFAULT, which is byte-identical to plain
    row-major NxM -- so a weight declared as "shape (1024, 768)" with no .transpose() reads
    correctly into a (768, 1024) numpy array using ordinary C-order reshape
  - .transpose() flips this: the declared (8, 2048) output layer, WITH .transpose() applied in
    this checkpoint's save_format, reads directly as genuine row-major (8, 2048)

SavedFormat entries used for this specific checkpoint (confirmed from the actual training
script, not assumed):
    l0w: (1024, 768), no transpose  -> reads as (768, 1024)
    l0b: (1024,)
    l1w: (8, 2048), .transpose()    -> reads as (8, 2048) directly
    l1b: (8,)
"""

import numpy as np

L0_HIDDEN = 1024
NUM_INPUTS = 768
NUM_BUCKETS = 8
L1_INPUT = 2 * L0_HIDDEN  # concatenated dual-perspective accumulator


def load_raw_bin(path: str) -> dict:
    data = np.fromfile(path, dtype="<f4")  # little-endian float32

    expected_total = (
        NUM_INPUTS * L0_HIDDEN  # l0w
        + L0_HIDDEN  # l0b
        + NUM_BUCKETS * L1_INPUT  # l1w
        + NUM_BUCKETS  # l1b
    )
    if data.size != expected_total:
        raise ValueError(
            f"raw.bin has {data.size} values, expected {expected_total} for a "
            f"768->{L0_HIDDEN}x2->{NUM_BUCKETS} architecture. Either this checkpoint uses a "
            f"different hidden size / bucket count, or the SavedFormat entries have changed -- "
            f"do not trust this loader on a file that fails this check."
        )

    offset = 0
    l0w = data[offset : offset + NUM_INPUTS * L0_HIDDEN].reshape(NUM_INPUTS, L0_HIDDEN)
    offset += NUM_INPUTS * L0_HIDDEN

    l0b = data[offset : offset + L0_HIDDEN]
    offset += L0_HIDDEN

    l1w = data[offset : offset + NUM_BUCKETS * L1_INPUT].reshape(NUM_BUCKETS, L1_INPUT)
    offset += NUM_BUCKETS * L1_INPUT

    l1b = data[offset : offset + NUM_BUCKETS]
    offset += NUM_BUCKETS

    assert offset == data.size

    return {
        "l0w": l0w.astype(np.float32),  # (768, 1024) -- feature index -> hidden weights row
        "l0b": l0b.astype(np.float32),  # (1024,)
        "l1w": l1w.astype(np.float32),  # (8, 2048) -- bucket -> weights over concat(us, them)
        "l1b": l1b.astype(np.float32),  # (8,)
    }
