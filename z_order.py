import numpy as np


def _spread_bits_3d(v):
    """
    Spreads the bits of a 21-bit integer by inserting two zeros between each bit.
    """
    v = np.asarray(v, dtype=np.uint64)

    # Enforce 21-bit limit to prevent overflow when interleaving 3 coordinates
    v = v & 0x1FFFFF

    # 3D Magic Numbers for 64-bit integers
    v = (v | (v << 32)) & 0x1F00000000FFFF
    v = (v | (v << 16)) & 0x1F0000FF0000FF
    v = (v | (v << 8)) & 0x100F00F00F00F00F
    v = (v | (v << 4)) & 0x10C30C30C30C30C3
    v = (v | (v << 2)) & 0x1249249249249249

    return v


def z_order_3d(x, y, z) -> np.ndarray:
    """
    Computes the 3D Z-order (Morton code) using magic numbers (no loops).
    Assumes inputs are arrays of non-negative integers.
    """
    x_spread = _spread_bits_3d(x)
    y_spread = _spread_bits_3d(y)
    z_spread = _spread_bits_3d(z)

    # Combine the spread bits:
    # x gets positions 0, 3, 6...
    # y gets positions 1, 4, 7...
    # z gets positions 2, 5, 8...
    return x_spread | (y_spread << 1) | (z_spread << 2)
