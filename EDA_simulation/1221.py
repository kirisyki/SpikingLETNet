def _format_scale_bits(scale, width: int) -> str:
    if isinstance(scale, (int, np.integer)):
        if scale < 0 or scale >= (1 << width):
            raise ValueError(f"scale 越界：需要 0 <= scale < 2^{width}，但得到 {scale}")
        return format(int(scale), f"0{width}b")

    if isinstance(scale, str):
        if len(scale) != width or any(c not in "01" for c in scale):
            raise ValueError(f"scale(bitstring) 必须是长度为 {width} 的 01 字符串")
        return scale

    raise TypeError("scale 必须是 int 或长度为 width 的二进制字符串")


def _to_twos_complement_bits_signed(value, width: int) -> str:
    
    v = int(value)
    min_v = -(1 << (width - 1))
    max_v = (1 << (width - 1)) - 1
    if v < min_v or v > max_v:
        raise ValueError(f"value 越界：有符号 {width}bit 需要 [{min_v}, {max_v}]，但得到 {v}")

    mask = (1 << width) - 1
    return format(v & mask, f"0{width}b")


def _transform_weight_3x3_block(W3: np.ndarray, bit_width: int = 4, scale=0) -> np.ndarray:
    if W3.shape != (3, 3):
        raise ValueError(f"_transform_weight_3x3_block 需要 3x3 矩阵，当前为 {W3.shape}")

    # row2 高 28bit 放 scale => 总长度 32bit => bit_width 必须为 4
    if bit_width != 4:
        raise ValueError(f"当前需求 scale 固定 28bit, 因此 bit_width 必须为 4: 但收到 bit_width={bit_width}")

    flat = W3.reshape(-1)

    # 新要求：每个元素是 4bit 有符号数，范围 [-8, 7]
    min_w = -(1 << (bit_width - 1))
    max_w = (1 << (bit_width - 1)) - 1
    if np.any(flat < min_w) or np.any(flat > max_w):
        raise ValueError(
            f"transform_weight 要求每个元素能用 {bit_width}bit 有符号表示，"
            f"即范围在 [{min_w}, {max_w}]，但检测到越界值。"
        )

    # row1: flat[7..0]，每个4bit拼成 32bit（补码）
    indices_row1 = [7, 6, 5, 4, 3, 2, 1, 0]
    bits_row1 = ''.join(_to_twos_complement_bits_signed(flat[idx], bit_width) for idx in indices_row1)

    # row2: 高28bit=scale（无符号），低4bit=flat[8]（补码）
    bits_elem9 = _to_twos_complement_bits_signed(flat[8], bit_width)  # 4bit
    scale_bits = _format_scale_bits(scale, 28)                        # 28bit
    bits_row2 = scale_bits + bits_elem9

    return np.array([bits_row1, bits_row2], dtype=object)


def transform_weight(W: np.ndarray, bit_width: int = 4, scale=0) -> np.ndarray:
    if W.shape == (3, 3):
        return _transform_weight_3x3_block(W, bit_width=bit_width, scale=scale)

    elif W.shape == (6, 6):
        block1 = W[0:3, 0:3]
        block2 = W[0:3, 3:6]
        block3 = W[3:6, 0:3]
        block4 = W[3:6, 3:6]

        outs = []
        for blk in (block1, block2, block3, block4):
            outs.append(_transform_weight_3x3_block(blk, bit_width=bit_width, scale=scale))

        # 4 个块，每块 (2,) -> (8,)
        return np.concatenate(outs, axis=0)

    else:
        raise ValueError(
            f"transform_weight 目前只支持 3x3 或 6x6 输入矩阵，当前形状为 {W.shape}"
        )
