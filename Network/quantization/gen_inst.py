import math
import subprocess
import os
import time
import re

from collections import deque
from typing import List, Dict
import numpy as np
import struct
import serial
import serial.tools.list_ports
import binascii
from binascii import b2a_hex, a2b_hex
import threading

#========================================================================================

def pad_rows_to_Z_4T_8T_3_3(A: np.ndarray) -> np.ndarray:
    # 获取 A 的行数 Y 和列数 X
    Y, X = A.shape

    # 定义 Z = 8 + 6a 的参数
    base = 8   
    step = 6   

    Z = base
    while Y > Z:
        Z += step

    Y_prime = Z  # 补齐后的目标行数

    # 创建一个全 0 的矩阵 B1，行数为 Y_prime，列数 X 相同
    B1 = np.zeros((Y_prime, X), dtype=A.dtype)

    # 把原矩阵 A 的内容复制到 B1 的前 Y 行中
    B1[:Y, :] = A

    return B1


def groups_transpose_4T_8T_3_3(A: np.ndarray) -> np.ndarray:
    # 获取行数和列数
    num_rows, num_cols = A.shape  # (Y', X)

    # 计算大组数量
    num_groups = (num_rows - 2) // 6

    blocks = []  # 用来保存每一组的 (X, 8) block

    # 逐组处理
    for g in range(num_groups):

        start_row = 6 * g

        group_rows = A[start_row:start_row + 8, :]

        reversed_rows = group_rows[::-1, :]

        block = reversed_rows.T

        blocks.append(block)

    B2 = np.vstack(blocks)

    return B2


def to_binary_matrix_4T_8T_3_3(M: np.ndarray, bit_width: int = 4) -> np.ndarray:
    # 使用 np.vectorize 给每个元素套一层格式化函数
    formatter = np.vectorize(lambda x: format(int(x), f'0{bit_width}b'))

    bin_M = formatter(M)

    return bin_M


def transform_two_stage_4T_8T_3_3(
    A: np.ndarray,
    to_binary: bool = False,
    bit_width: int = 8
):

    # 第一步：补行数
    B1 = pad_rows_to_Z_4T_8T_3_3(A)

    # 根据补零后的行数计算 a_out
    Y_prime, _ = B1.shape
    # 理论上 (Y_prime - 2) 必然是 6 的倍数
    a_out = (Y_prime - 2) // 6

    # 第二步：分组 + 转置
    C = groups_transpose_4T_8T_3_3(B1)

    # 如果需要二进制形式，在这里做第三步转换
    if to_binary:
        C_bin = to_binary_matrix_4T_8T_3_3(C, bit_width=bit_width)
        return C_bin, a_out

    return C, a_out


# ==========================================================================


def pad_rows_to_Z_4T_8T_1_3(A: np.ndarray) -> np.ndarray:
    # 获取 A 的行数 Y 和列数 X
    Y, X = A.shape

    # 直接算出 >=Y 的最小 8 的倍数
    if Y % 8 == 0:
        Y_prime = Y
    else:
        Y_prime = (Y // 8 + 1) * 8

    # 创建一个全 0 的矩阵 B1，行数为 Y_prime，列数 X 相同
    B1 = np.zeros((Y_prime, X), dtype=A.dtype)

    # 把原矩阵 A 的内容复制到 B1 的前 Y 行中
    B1[:Y, :] = A

    return B1


def groups_transpose_4T_8T_1_3(A: np.ndarray) -> np.ndarray:
    # 获取行数和列数
    num_rows, num_cols = A.shape  # (Y', X)

    # 第一阶段保证 num_rows 为 8 的倍数
    num_groups = num_rows // 8

    blocks = []  # 每组得到一个 (X, 8) block

    for g in range(num_groups):
        start_row = 8 * g
        group_rows = A[start_row:start_row + 8, :]     # (8, X)
        reversed_rows = group_rows[::-1, :]            # 行倒序
        block = reversed_rows.T                        # (X, 8)
        blocks.append(block)

    B2 = np.vstack(blocks)  # (num_groups * X, 8)
    return B2


def to_binary_matrix_4T_8T_1_3(M: np.ndarray, bit_width: int = 4) -> np.ndarray:
    formatter = np.vectorize(lambda x: format(int(x), f"0{bit_width}b"))
    return formatter(M)


def transform_two_stage_4T_8T_1_3(
    A: np.ndarray,
    to_binary: bool = False,
    bit_width: int = 8
):
    """
    两级变换：
      1) 行补零到 8 的倍数
      2) 每 8 行一组：行倒序 + 转置 + 纵向拼接

    新增输出 a_out：
      a_out 为满足 Y ≤ 8 + 8(a-1) 的最小正整数 a
      等价于：a_out = ceil(Y/8) = (Y+7)//8
    """
    # 用输入矩阵的行数 Y 来定义 a_out
    Y, X = A.shape
    a_out = (Y + 7) // 8

    # 第一步：补行数到 8 的倍数
    B1 = pad_rows_to_Z_4T_8T_1_3(A)

    # 第二步：分组 + 翻转 + 转置
    C = groups_transpose_4T_8T_1_3(B1)

    if to_binary:
        C_bin = to_binary_matrix_4T_8T_1_3(C, bit_width=bit_width)
        return C_bin, a_out

    return C, a_out




# ==========================================================================


def pad_rows_to_Z_4T_8T_3_1(A: np.ndarray) -> np.ndarray:
    # 获取 A 的行数 Y 和列数 X
    Y, X = A.shape

    # 直接算出 >=Y 的最小 8 的倍数
    if Y % 8 == 0:
        Y_prime = Y
    else:
        Y_prime = (Y // 8 + 1) * 8

    # 创建一个全 0 的矩阵 B1，行数为 Y_prime，列数 X 相同
    B1 = np.zeros((Y_prime, X), dtype=A.dtype)

    # 把原矩阵 A 的内容复制到 B1 的前 Y 行中
    B1[:Y, :] = A

    return B1


def groups_transpose_4T_8T_3_1(A: np.ndarray) -> np.ndarray:
    # 获取行数和列数
    num_rows, num_cols = A.shape  # (Y', X)

    # 第一阶段保证 num_rows 为 8 的倍数
    num_groups = num_rows // 8

    blocks = []  # 每组得到一个 (num_cols, 8) block

    for g in range(num_groups):
        start_row = 8 * g
        group_rows = A[start_row:start_row + 8, :]   # (8, num_cols)
        reversed_rows = group_rows[::-1, :]          # 行倒序
        block = reversed_rows.T                      # (num_cols, 8)
        blocks.append(block)

    B2 = np.vstack(blocks)  # (num_groups * num_cols, 8)
    return B2


def to_binary_matrix_4T_8T_3_1(M: np.ndarray, bit_width: int = 4) -> np.ndarray:
    formatter = np.vectorize(lambda x: format(int(x), f"0{bit_width}b"))
    return formatter(M)


def transform_two_stage_4T_8T_3_1(
    A: np.ndarray,
    to_binary: bool = False,
    bit_width: int = 8
):
    """
    新要求：输出 a_out，按输入矩阵 A 的列数 X 定义：
      X ≤ 8 + 8(a-1), a 为正整数
    等价于：
      a_out = ceil(X/8) = (X + 7)//8
    """
    # a_out 只按输入矩阵的列数 X 计算（与后续转置/补零无关）
    _, X = A.shape
    a_out = (X + 7) // 8

    # 先做一次整体转置（你现有代码的“新要求”）
    A_T = A.T

    # 第一步：对转置后的矩阵补行数到 8 的倍数
    B1 = pad_rows_to_Z_4T_8T_3_1(A_T)

    # 第二步：分组 + 翻转 + 转置（无重叠大组）
    C = groups_transpose_4T_8T_3_1(B1)

    # 如果需要二进制形式
    if to_binary:
        C_bin = to_binary_matrix_4T_8T_3_1(C, bit_width=bit_width)
        return C_bin, a_out

    # 否则返回整数矩阵
    return C, a_out



# ==========================================================================


import numpy as np


def pad_rows_to_Z_dilated_1_2(A: np.ndarray) -> tuple[np.ndarray, int]:

    if A.ndim != 2:
        raise ValueError(f"A 必须是 2D 矩阵，但收到 A.ndim={A.ndim}")

    Y, X = A.shape
    if Y <= 0 or X <= 0:
        raise ValueError(f"A 形状非法：Y={Y}, X={X}")

    a_out = (Y + 7) // 8
    Y_prime = 8 * a_out

    A_pad = np.zeros((Y_prime, X), dtype=A.dtype)
    A_pad[:Y, :] = A
    return A_pad, a_out


def pad_cols_to_even_after_rowpad_dilated_1_2(A_pad: np.ndarray) -> np.ndarray:
  
    if A_pad.ndim != 2:
        raise ValueError(f"A_pad 必须是 2D 矩阵，但收到 A_pad.ndim={A_pad.ndim}")

    Yp, X = A_pad.shape
    if X % 2 == 0:
        return A_pad

    A_pad2 = np.zeros((Yp, X + 1), dtype=A_pad.dtype)
    A_pad2[:, :X] = A_pad
    return A_pad2


def groups_transpose_dilated_1_2(A_pad: np.ndarray) -> np.ndarray:
  
    if A_pad.ndim != 2:
        raise ValueError(f"A_pad 必须是 2D 矩阵，但收到 A_pad.ndim={A_pad.ndim}")

    Yp, X = A_pad.shape
    if Yp % 8 != 0:
        raise ValueError(f"补零后行数必须是 8 的倍数，但收到 Y'={Yp}")

    num_groups = Yp // 8
    odd_cols = list(range(0, X, 2))   # 0,2,4,... -> 1-based 1,3,5...
    even_cols = list(range(1, X, 2))  # 1,3,5,... -> 1-based 2,4,6...

    out_rows = []

    for g in range(num_groups):
        grp = A_pad[g * 8:(g + 1) * 8, :]
        for c in odd_cols:
            out_rows.append(grp[:, c][::-1])

    for g in range(num_groups):
        grp = A_pad[g * 8:(g + 1) * 8, :]
        for c in even_cols:
            out_rows.append(grp[:, c][::-1])

    return np.array(out_rows, dtype=A_pad.dtype)


def to_binary_matrix_dilated_1_2(M: np.ndarray, bit_width: int = 4) -> np.ndarray:

    formatter = np.vectorize(lambda x: format(int(x), f"0{bit_width}b"))
    return formatter(M)


def transform_two_stage_dilated_1_2(A: np.ndarray, to_binary: bool = False, bit_width: int = 8):
   
    # 1) 先补行
    A_pad, a_out = pad_rows_to_Z_dilated_1_2(A)

    # 2) 再判断列数：奇数则补1列0
    A_pad = pad_cols_to_even_after_rowpad_dilated_1_2(A_pad)

    # 3) 原来的分组变换
    C = groups_transpose_dilated_1_2(A_pad)

    # 4) 可选：转二进制
    if to_binary:
        return to_binary_matrix_dilated_1_2(C, bit_width=bit_width), a_out
    return C, 2*a_out


# ==========================================================================


import numpy as np


def pad_rows_to_Z_dilated_2_1(A: np.ndarray) -> np.ndarray:
  
    Y, X = A.shape
    if Y <= 0 or X <= 0:
        raise ValueError(f"A 形状非法：Y={Y}, X={X}")

    a = (Y + 7) // 8
    Y_prime = 8 * a

    B1 = np.zeros((Y_prime, X), dtype=A.dtype)
    B1[:Y, :] = A
    return B1


def pad_cols_to_even_after_rowpad_dilated_2_1(B1: np.ndarray) -> np.ndarray:
 
    if B1.ndim != 2:
        raise ValueError(f"B1 必须是 2D 矩阵，但收到 B1.ndim={B1.ndim}")

    num_rows, num_cols = B1.shape
    if num_cols % 2 == 0:
        return B1

    B1_even = np.zeros((num_rows, num_cols + 1), dtype=B1.dtype)
    B1_even[:, :num_cols] = B1
    return B1_even


def groups_transpose_dilated_2_1(A: np.ndarray) -> np.ndarray:
 
    num_rows, num_cols = A.shape

    if num_rows % 8 != 0:
        raise ValueError(f"行数 {num_rows} 必须是 8 的倍数（已补齐到 8a），但当前不是。")

    num_groups = num_rows // 8
    odd_cols = list(range(0, num_cols, 2))   # 0,2,4,... -> 1-based 1,3,5...
    even_cols = list(range(1, num_cols, 2))  # 1,3,5,... -> 1-based 2,4,6...

    rows_list = []

    # 先：奇数列（按组顺序）
    for g in range(num_groups):
        grp = A[g * 8:(g + 1) * 8, :]
        for c in odd_cols:
            rows_list.append(grp[:, c][::-1])

    # 再：偶数列（按组顺序）
    for g in range(num_groups):
        grp = A[g * 8:(g + 1) * 8, :]
        for c in even_cols:
            rows_list.append(grp[:, c][::-1])

    return np.vstack(rows_list) if rows_list else np.zeros((0, 8), dtype=A.dtype)


def to_binary_matrix_dilated_2_1(M: np.ndarray, bit_width: int = 4) -> np.ndarray:
    formatter = np.vectorize(lambda x: format(int(x), f"0{bit_width}b"))
    return formatter(M)


def transform_two_stage_dilated_2_1(
    A: np.ndarray,
    to_binary: bool = False,
    bit_width: int = 8
):

    if A.ndim != 2:
        raise ValueError(f"A 必须是二维矩阵，但收到 A.ndim={A.ndim}")

    Y, X = A.shape
    if Y <= 0 or X <= 0:
        raise ValueError(f"A 形状非法：Y={Y}, X={X}")

    # a_out 由转置后的行数 (= 原 X) 决定
    a_out = (X + 7) // 8

    # 0) 转置
    A_T = A.T  # shape = (X, Y)

    # 1) 补行到 8a
    B1 = pad_rows_to_Z_dilated_2_1(A_T)  # shape = (8*a_out, Y)

    # 1.5) ✅ 补行后检查列数，奇数则补1列0
    B1 = pad_cols_to_even_after_rowpad_dilated_2_1(B1)  # shape = (8*a_out, Y or Y+1)

    # 2) 分组变换
    C = groups_transpose_dilated_2_1(B1)

    # 3) 可选：转二进制
    if to_binary:
        return to_binary_matrix_dilated_2_1(C, bit_width=bit_width), a_out

    return C, 2*a_out


# ==========================================================================


import numpy as np


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


# ==========================================================================


def pad_to_4a_gemm(A1: np.ndarray, A2: np.ndarray):

    # 读取原始形状
    Y1, X1 = A1.shape    # A1: (Y1, X1)
    X2, Y2 = A2.shape    # A2: (X2, Y2)

    # 基本维度检查：只要求公共维度一致
    if X1 != X2:
        raise ValueError(
            f"维度不匹配：A1 的列数为 {X1}，A2 的行数为 {X2}，两者必须相等。"
        )

    X = X1  # 公共维度

    # 计算需要补到的 4 的倍数 X_pad
    if X % 4 == 0:
        X_pad = X
    else:
        X_pad = (X // 4 + 1) * 4   # 向上取到最近的 4 的倍数

    a = X_pad // 4

    # 如果本身就满足 X = 4a，则无需补零
    if X_pad == X:
        return A1, A2, a

    # ===== 需要补零的情况 =====
    A1_pad = np.zeros((Y1, X_pad), dtype=A1.dtype)
    A1_pad[:, :X] = A1

    A2_pad = np.zeros((X_pad, Y2), dtype=A2.dtype)
    A2_pad[:X, :] = A2

    return A1_pad, A2_pad, a


def transform_A1_A2_to_B_gemm(
    A1: np.ndarray,
    A2: np.ndarray,
    do_pad: bool = True
) -> np.ndarray:

    # 1) 按需要补零，使 X 变成 4a
    if do_pad:
        A1_pad, A2_pad, a = pad_to_4a_gemm(A1, A2)
    else:
        A1_pad, A2_pad = A1, A2
        Y1, X1 = A1_pad.shape
        X2, Y2 = A2_pad.shape

        # 仅要求公共维度一致
        if X1 != X2:
            raise ValueError(
                f"do_pad=False 时，仍然要求 A1 的列数 = A2 的行数，"
                f"当前 A1.shape={A1_pad.shape}, A2.shape={A2_pad.shape}"
            )
        if X1 % 4 != 0:
            raise ValueError(
                "do_pad=False，但公共维度 X 不是 4 的倍数，无法按 4 个一组分块。"
            )
        a = X1 // 4

    # 重新读取补零后的形状
    Y1, X = A1_pad.shape             
    X_check, Y2 = A2_pad.shape       

    if X_check != X:
        raise RuntimeError(
            f"内部错误：补零后 A1_pad、A2_pad 的公共维度不一致，"
            f"A1_pad.shape={A1_pad.shape}, A2_pad.shape={A2_pad.shape}"
        )

    # 输出数据类型：根据 A1_pad 和 A2_pad 的 dtype 推导
    out_dtype = np.result_type(A1_pad.dtype, A2_pad.dtype)

    # 原始输出 B_raw 的行数 / 列数
    blocks_per_pair = a         
    num_pairs = Y1 * Y2        
    num_rows_B_raw = num_pairs * blocks_per_pair
    num_cols_B = 8             

    B_raw = np.zeros((num_rows_B_raw, num_cols_B), dtype=out_dtype)
    row_idx = 0

    # 2) 遍历 A1 的每一行 i
    for i in range(Y1):
        # 3) 遍历 A2 的每一列 j
        for j in range(Y2):
            # 4) 在公共维度 X 上按 4 个一组分块
            for k in range(blocks_per_pair):
                start = 4 * k
                end = start + 4

                # A1 第 i 行的 4 个元素
                A1_block = A1_pad[i, start:end]    

                # A2 第 j 列的 4 个元素（在行方向）
                A2_block = A2_pad[start:end, j]   

                # 写入 B_raw 当前行
                B_raw[row_idx, 0:4] = A1_block
                B_raw[row_idx, 4:8] = A2_block
                row_idx += 1

    if row_idx != num_rows_B_raw:
        raise RuntimeError(
            f"内部错误：理论应写 {num_rows_B_raw} 行，但实际写入 {row_idx} 行。"
        )

    # ===== 按你的约定，在最前面加两行 0 =====
    B = np.zeros((num_rows_B_raw + 2, num_cols_B), dtype=out_dtype)
    B[2:, :] = B_raw

    return B


def to_binary_matrix_gemm(M: np.ndarray, bit_width: int = 4) -> np.ndarray:
    formatter = np.vectorize(lambda x: format(int(x), f'0{bit_width}b'))
    bin_M = formatter(M)
    return bin_M


def transform_A1_A2_to_B_with_binary_gemm(
    A1: np.ndarray,
    A2: np.ndarray,
    bit_width: int = 4,
    do_pad: bool = True
):
    # 主变换 + 二进制
    B_dec = transform_A1_A2_to_B_gemm(A1, A2, do_pad=do_pad)
    B_bin = to_binary_matrix_gemm(B_dec, bit_width=bit_width)

    # 新增：根据原始 A1/A2 的形状计算 a_out
    Y1, _ = A1.shape
    _, Y2 = A2.shape
    a_out = Y1 * Y2

    return B_dec, B_bin, a_out


# ==========================================================================


def compute_group_starts_6_6(first_row: int,
                             last_row_inclusive: int,
                             group_height: int = 8,
                             stride: int = 6) -> list:

    starts = []
    s = first_row
    while s + group_height - 1 <= last_row_inclusive:
        starts.append(s)
        s += stride
    return starts


def pad_rows_to_8_3_6a_6_6(A: np.ndarray) -> np.ndarray:

    Y, X = A.shape

    if Y <= 0:
        raise ValueError(f"pad_rows_to_8_3_6a_6_6 要求 A 至少有 1 行，当前 Y = {Y}")

    # 最小从 17 开始：17, 23, 29, ...
    Y_prime = 17
    while Y_prime < Y:
        Y_prime += 6

    if Y_prime == Y:
        # 已经满足 11 + 6*(a-1)，不需要补
        return A

    B = np.zeros((Y_prime, X), dtype=A.dtype)
    B[:Y, :] = A
    return B


def transform_one_stage_6_6(A: np.ndarray,
                            row_starts: list,
                            col_indices: list):


    num_rows, num_cols = A.shape
    group_height = 8

    # ===== 计算 a_out =====
    if num_rows < 11 or (num_rows - 11) % 6 != 0:
        raise ValueError(
            f"transform_one_stage_6_6 要求行数 Y 满足 Y = 11 + 6*(a-1)，"
            f"当前 Y = {num_rows}"
        )
    a_out = (num_rows - 11) // 6 + 1   # a = (Y - 11)/6 + 1

    all_rows = []

    for s in row_starts:
        if s < 0 or s + group_height > num_rows:
            raise RuntimeError(
                f"大组起始行 s={s} 越界，矩阵共有 {num_rows} 行。"
            )

        # 从下到上：s+7, ..., s
        rows_bottom_to_top = list(range(s + group_height - 1, s - 1, -1))

        for c in col_indices:
            if c < 0 or c >= num_cols:
                raise RuntimeError(
                    f"列下标 c={c} 越界，矩阵共有 {num_cols} 列。"
                )
            row_vals = A[rows_bottom_to_top, c]  # (8,)
            all_rows.append(row_vals)

    if not all_rows:
        # 发生的话说明行数太小，无法形成任何大组
        B_stage = np.zeros((0, 8), dtype=A.dtype)
        return B_stage, a_out

    B_stage = np.vstack(all_rows)
    return B_stage, a_out


def transform_four_stages_6_6(A: np.ndarray):

    # 先在行方向补零到满足 11 + 6*(a-1)
    A_pad = pad_rows_to_8_3_6a_6_6(A)
    Y, X = A_pad.shape

    if X <= 3:
        raise ValueError("列数 X 必须大于 3，否则去掉 3 列后无法进行变换。")

    # ---- 计算 a_out（对补零后的 Y 来算）----
    if Y < 11 or (Y - 11) % 6 != 0:
        # 理论上 pad_rows_to_8_3_6a_6_6 已经保证不会发生
        raise ValueError(
            f"transform_four_stages_6_6 内部错误：补零后行数 Y={Y} "
            f"不满足 Y = 11 + 6*(a-1)"
        )
    a_out = (Y - 11) // 6 + 1   # a = (Y - 11)/6 + 1

    # ===== 阶段 1 & 2：行 0 ~ Y-4 =====
    last_row_stage12 = Y - 4
    if last_row_stage12 < 7:
        # 理论上补到 Y'≥17 后不会发生，仅作防御性检查
        raise ValueError("行数不足以支撑阶段 1/2 的一个大组。")

    row_starts_stage12 = compute_group_starts_6_6(
        first_row=0,
        last_row_inclusive=last_row_stage12,
        group_height=8,
        stride=6
    )

    # 阶段 1：去掉最右 3 列
    cols_stage1 = list(range(0, X - 3))
    B1, a_out_stage1 = transform_one_stage_6_6(A_pad, row_starts_stage12, cols_stage1)

    # 阶段 2：去掉最左 3 列
    cols_stage2 = list(range(3, X))
    B2, a_out_stage2 = transform_one_stage_6_6(A_pad, row_starts_stage12, cols_stage2)

    # ===== 阶段 3 & 4：行 3 ~ Y-1 =====
    first_row_stage34 = 3
    last_row_stage34 = Y - 1
    if last_row_stage34 - first_row_stage34 + 1 < 8:
        raise ValueError("行数不足以支撑阶段 3/4 的一个大组。")

    row_starts_stage34 = compute_group_starts_6_6(
        first_row=first_row_stage34,
        last_row_inclusive=last_row_stage34,
        group_height=8,
        stride=6
    )

    # 阶段 3：去掉最右 3 列
    cols_stage3 = list(range(0, X - 3))
    B3, a_out_stage3 = transform_one_stage_6_6(A_pad, row_starts_stage34, cols_stage3)

    # 阶段 4：去掉最左 3 列
    cols_stage4 = list(range(3, X))
    B4, a_out_stage4 = transform_one_stage_6_6(A_pad, row_starts_stage34, cols_stage4)

    # ---- 可选：检查四个 stage 的 a_out 是否一致（调试用）----
    if not (a_out_stage1 == a_out_stage2 == a_out_stage3 == a_out_stage4 == a_out):
        raise RuntimeError(
            "transform_four_stages_6_6: 各阶段计算得到的 a_out 不一致，"
            f"a_out={a_out}, stages=({a_out_stage1},{a_out_stage2},"
            f"{a_out_stage3},{a_out_stage4})"
        )

    # 按 1→2→3→4 顺序拼接
    B_all = np.vstack([B1, B2, B3, B4])

    # 关键：返回两个量 (B_all, a_out)，和 transform_all 中的解包完全一致
    return B_all, a_out*4


# ==========================================================================


import numpy as np

def transform_all(
    A: np.ndarray,
    A2: np.ndarray,
    W: np.ndarray,
    mode,
    pe_mode_in,
    step_mode_in,
    enable_in,
    to_binary: bool = False,
    bit_width: int = 8,
    scale=0,
):

    # ---------- 0. 统一解析 enable_in / enable_out ----------
    enable_out = 1 if int(enable_in) == 1 else 0

    # ---------- 1. 统一把 mode 转成整数（即使 enable_out=0 也要求 mode 合法） ----------
    if isinstance(mode, str):
        mode_str = mode.strip()
        if not mode_str.isdigit():
            raise ValueError(f"mode 字符串必须是数字 '0'~'6'，当前为: {mode}")
        mode_int = int(mode_str)
    elif isinstance(mode, bool):
        mode_int = int(mode)
    else:
        mode_int = int(mode)

    if mode_int not in (0, 1, 2, 3, 4, 5, 6):
        raise ValueError(
            f"当前 transform_all 只支持 mode = 0/1/2/3/4/5/6，收到的 mode = {mode_int}。"
        )

    # ========= 1.1 统计输入矩阵 A 的尺寸 =========
    Y, X = A.shape  #   Y: 行数, X: 列数

    # ---------- 2. 生成 write_depth_out / pe_mode_out / step_mode_out / kernel_size_out ----------

    # 2.1 新规则：write_depth_out
    if mode_int in (0, 2):
        write_depth_out = X

    elif mode_int == 3:
        write_depth_out = (X + 1) // 2

    elif mode_int == 1:
        write_depth_out = Y

    elif mode_int == 4:
        write_depth_out = (Y + 1) // 2

    elif mode_int == 5:
        if X <= 0:
            raise ValueError(f"mode=5 时要求 X 为正整数，当前 X = {X}")
        if X % 4 == 0:
            X_pad = X
        else:
            X_pad = (X // 4 + 1) * 4
        write_depth_out = X_pad // 4

    elif mode_int == 6:
        if X < 4:
            raise ValueError(f"mode=6 时要求 X >= 4，以便 X-3 为正，当前 X = {X}")
        write_depth_out = X - 3

    # 2.2 pe_mode_out = pe_mode_in（检查范围 0/1/2）
    pe_mode_out = int(pe_mode_in)
    if pe_mode_out not in (0, 1, 2):
        raise ValueError(f"pe_mode_in 只能是 0/1/2，当前为 {pe_mode_in}")

    # 2.3 step_mode_out = step_mode_in（检查范围 4/8/12）
    step_mode_out = int(step_mode_in)
    if step_mode_out not in (4, 8, 12):
        raise ValueError(f"step_mode_in 只能是 4/8/12，当前为 {step_mode_in}")

    # 2.4 kernel_size_out
    if mode_int in (0, 3, 4, 5, 6):
        kernel_size_out = "3*3"
    else:
        kernel_size_out = "3*1"

    # ---------- 3. 如果 enable_out=0：不做任何运算，直接返回 ----------
    if enable_out == 0:
        a_out = None
        return (
            None,          # C_main
            None,          # W_bin
            write_depth_out,
            pe_mode_out,
            step_mode_out,
            kernel_size_out,
            enable_out,
            a_out
        )

    # ---------- 4. enable_out=1：按 mode 对 W 的尺寸做检查 ----------
    if mode_int in (1, 2, 3, 4):
        if W.shape != (3, 3):
            raise ValueError(f"mode={mode_int} 时，W 只允许是 3x3，当前形状为 {W.shape}")
    elif mode_int == 6:
        if W.shape != (6, 6):
            raise ValueError(f"mode=6 时，W 只允许是 6x6，当前形状为 {W.shape}")
    elif mode_int == 0:
        if W.shape not in ((3, 3), (6, 6)):
            raise ValueError(f"mode=0 时，W 只允许是 3x3 或 6x6，当前形状为 {W.shape}")
    elif mode_int == 5:
        if W.shape not in ((3, 3), (6, 6)):
            raise ValueError(f"mode=5 时，W 只允许是 3x3 或 6x6，当前形状为 {W.shape}")

    # ---------- 5. 主路径：统一得到十进制矩阵 C_dec 以及 a_out ----------
    if mode_int == 0:
        C_dec, a_out = transform_two_stage_4T_8T_3_3(A, to_binary=False, bit_width=bit_width)

    elif mode_int == 1:
        C_dec, a_out = transform_two_stage_4T_8T_3_1(A, to_binary=False, bit_width=bit_width)

    elif mode_int == 2:
        C_dec, a_out = transform_two_stage_4T_8T_1_3(A, to_binary=False, bit_width=bit_width)

    elif mode_int == 3:
        C_dec, a_out = transform_two_stage_dilated_1_2(A, to_binary=False, bit_width=bit_width)

    elif mode_int == 4:
        C_dec, a_out = transform_two_stage_dilated_2_1(A, to_binary=False, bit_width=bit_width)

    elif mode_int == 5:
        B_dec, _B_bin, a_out = transform_A1_A2_to_B_with_binary_gemm(
            A1=A, A2=A2, bit_width=bit_width, do_pad=True
        )
        C_dec = B_dec

    elif mode_int == 6:
        C_dec, a_out = transform_four_stages_6_6(A)

    # ---------- 6. 权重路径：交给 transform_weight（新增 scale 透传） ----------
    W_bin = transform_weight(W, bit_width=4, scale=scale)

    # ---------- 7. 根据 to_binary 决定主输出形式（✅按 pe_mode 区分有符号/无符号） ----------
    if not to_binary:
        C_main = C_dec
    else:
        if not isinstance(bit_width, (int, np.integer)) or int(bit_width) <= 0:
            raise ValueError(f"bit_width 必须是正整数，但收到: {bit_width}")
        bw = int(bit_width)
        mask = (1 << bw) - 1

        def _int_to_bin_signed(v) -> str:
            """有符号：按 two's complement 编码为 bw 位。"""
            iv = int(v)
            minv = -(1 << (bw - 1))
            maxv = (1 << (bw - 1)) - 1
            if iv < minv or iv > maxv:
                raise ValueError(f"有符号 {bw}bit 范围为 [{minv},{maxv}]，但收到 {iv}")
            u = iv & mask
            return format(u, f"0{bw}b")

        def _int_to_bin_unsigned(v) -> str:
            """无符号：0..2^bw-1。"""
            iv = int(v)
            if iv < 0 or iv > mask:
                raise ValueError(f"无符号 {bw}bit 范围为 [0,{mask}]，但收到 {iv}")
            return format(iv, f"0{bw}b")

        # ✅ 新规则：
        #   pe_mode_out==2(gemm)：有符号
        #   pe_mode_out==0(cnn) 或 1(snn)：无符号
        if pe_mode_out == 2:
            formatter = np.vectorize(_int_to_bin_signed)
        else:  # cnn/snn
            formatter = np.vectorize(_int_to_bin_unsigned)

        C_bin_chunks = formatter(C_dec)

        # 保持你原来的输出格式：每一行拼接成一个长 bit 串
        if C_bin_chunks.ndim == 1:
            C_main = C_bin_chunks.astype(str)
        else:
            C_main = np.array(["".join(row) for row in C_bin_chunks], dtype=str)

    return (
        C_main,
        W_bin,
        write_depth_out,
        pe_mode_out,
        step_mode_out,
        kernel_size_out,
        enable_out,
        a_out
    )



# ======================================================================================================


import numpy as np
import os
import glob

def transform_all_1(
    A_list, A2_list, W_list,
    kernel_size_in_list,
    dialation_in_list,
    step_mode_in_list,
    scale_list,
    pe_mode_in_list,
    kernel_num
):
    """
    输入参数顺序（最新）：
      A_list, A2_list, W_list,
      kernel_size_in_list, dialation_in_list, step_mode_in_list,
      scale_list, pe_mode_in_list,
      kernel_num

    dialation_in_list: list[str]，元素必须是 "1*2"/"2*1"/"1*1"
      "1*2" -> (1,2)
      "2*1" -> (2,1)
      "1*1" -> None

    kernel_size_in_list: list[str]，如 "3*3","3*1","1*3","6*6"
    step_mode_in_list: list[str]，"4"/"8"/"12"

    新规则（本次修改）：
      - pe_mode=gemm(2)：A 元素按 4bit two's complement 有符号解码（-8..7）
      - pe_mode=cnn(0)/snn(1)：A 元素按 4bit 无符号解码（0..15）
      - snn：在无符号解码后，再按 step_mode 映射表映射

    输出：
      - depth_list: list[int]
      - group_list: list[int]
      - group_max: int
    """

    # ========= kernel_num 检查 =========
    if not isinstance(kernel_num, (int, np.integer)):
        raise TypeError(f"kernel_num 必须是 int，但收到 {type(kernel_num)}: {kernel_num}")
    kernel_num = int(kernel_num)
    if kernel_num <= 0:
        raise ValueError(f"kernel_num 必须 > 0，但收到 {kernel_num}")

    # ========= scale_list 默认 =========
    if scale_list is None:
        scale_list = [0] * kernel_num

    def _check_min_len(name, lst):
        if not isinstance(lst, (list, tuple)):
            raise TypeError(f"{name} 必须是 list/tuple")
        if len(lst) < kernel_num:
            raise ValueError(f"{name} 长度必须 >= kernel_num={kernel_num}，但 {name}={len(lst)}")

    _check_min_len("A_list", A_list)
    _check_min_len("A2_list", A2_list)
    _check_min_len("W_list", W_list)
    _check_min_len("kernel_size_in_list", kernel_size_in_list)
    _check_min_len("dialation_in_list", dialation_in_list)
    _check_min_len("step_mode_in_list", step_mode_in_list)
    _check_min_len("scale_list", scale_list)
    _check_min_len("pe_mode_in_list", pe_mode_in_list)

    # =========================
    # 内部工具
    # =========================
    def _normalize_kernel_size_str(ks: str) -> str:
        if not isinstance(ks, str):
            raise TypeError(f"kernel_size_in 必须是 str（如 '3*3'），但收到 {type(ks)}: {ks}")
        s = ks.strip()
        parts = [p.strip() for p in s.split("*")]
        if len(parts) != 2:
            raise ValueError(f"kernel_size_in='{ks}' 格式错误，必须是 'x*y'")
        x = int(parts[0])
        y = int(parts[1])
        return f"{x}*{y}"

    def _normalize_step_mode_str(sm: str) -> str:
        if not isinstance(sm, str):
            raise TypeError(f"step_mode_in 必须是 str（'4'/'8'/'12'），但收到 {type(sm)}: {sm}")
        s = sm.strip()
        if s not in ("4", "8", "12"):
            raise ValueError(f"step_mode_in='{sm}' 不支持，只支持 '4'/'8'/'12'")
        return s

    def _normalize_step_mode_out_str(step_mode_out) -> str:
        return str(int(step_mode_out))

    def _normalize_dilation_str(di: str):
        if not isinstance(di, str):
            raise TypeError(f"dialation_in 必须是 str（'1*1'/'1*2'/'2*1'），但收到 {type(di)}: {di}")
        s = di.strip()
        parts = [p.strip() for p in s.split("*")]
        if len(parts) != 2:
            raise ValueError(f"dialation_in='{di}' 格式错误，必须是 'x*y'")
        x = int(parts[0])
        y = int(parts[1])

        if (x, y) == (1, 1):
            return "1*1", None
        if (x, y) == (1, 2):
            return "1*2", (1, 2)
        if (x, y) == (2, 1):
            return "2*1", (2, 1)

        raise ValueError(f"dialation_in='{di}' 不支持，只支持 '1*1'/'1*2'/'2*1'")

    def _compute_mode(pe_mode_in, dialation_call, kernel_size_in_std: str) -> int:
        pe = int(pe_mode_in)

        if pe == 2:  # GEMM
            return 5

        if dialation_call is not None:
            if tuple(dialation_call) == (1, 2):
                return 3
            if tuple(dialation_call) == (2, 1):
                return 4
            raise ValueError(f"dialation_in={dialation_call} 不支持，只支持 (1,2)/(2,1)/None")

        if kernel_size_in_std == "3*3":
            return 0
        if kernel_size_in_std == "3*1":
            return 1
        if kernel_size_in_std == "1*3":
            return 2
        if kernel_size_in_std == "6*6":
            return 6
        raise ValueError(f"kernel_size_in='{kernel_size_in_std}' 不支持")

    def _map_pe_mode_out(pe_mode_out) -> str:
        v = int(pe_mode_out)
        if v == 0:
            return "cnn"
        if v == 1:
            return "snn"
        if v == 2:
            return "gemm"
        raise ValueError(f"不支持的 pe_mode_out 数值: {pe_mode_out}（只支持 0/1/2）")

    def _normalize_pe_mode_in(pe_mode_in):
        if isinstance(pe_mode_in, (int, np.integer)):
            v = int(pe_mode_in)
            if v in (0, 1, 2):
                return v
        if isinstance(pe_mode_in, str):
            s = pe_mode_in.strip().lower()
            if s in ("ann", "cnn"):
                return 0
            if s == "snn":
                return 1
            if s == "gemm":
                return 2
        raise ValueError(f"不支持的 pe_mode_in={pe_mode_in}（请用 0/1/2 或 ann/snn/gemm）")

    # ========== SNN 预映射（int 版）==========
    _SNN_MAP_4  = {0: 0, 1: 1, 2: 3, 3: 7, 4: 15}
    _SNN_MAP_8  = {0: 0, 1: 1, 2: 3, 3: 2, 4: 6, 5: 7, 6: 5, 7: 4, 8: 12}
    _SNN_MAP_12 = {0: 0, 1: 1, 2: 3, 3: 2, 4: 6, 5: 7, 6: 5, 7: 4, 8: 12, 9: 13, 10: 15, 11: 14, 12: 10}

    # =========================
    # 4bit 二进制解析（无符号/有符号）
    # =========================
    def _parse_bin_str_4(x) -> int:
        """把 x 解析为 0..15 的无符号整数（原始 bit pattern），不做符号解释。"""
        if isinstance(x, (int, np.integer)):
            v = int(x)
            if 0 <= v <= 15:
                return v
            raise ValueError(f"A 元素十进制值超出 0..15: {v}")

        if isinstance(x, (str, np.str_)):
            s = str(x).strip()
            if s == "":
                raise ValueError("A 中出现空字符串 ''")
            if any(ch not in "01" for ch in s):
                raise ValueError(f"A 元素不是二进制字符串: {repr(s)}")
            if len(s) > 4:
                raise ValueError(f"A 元素超过 4bit: {repr(s)}")
            return int(s.zfill(4), 2)

        raise TypeError(f"A 元素类型不支持: {type(x)}")

    def _parse_sint_4(x) -> int:
        """
        把 x 解析为 4bit two's complement 有符号整数：-8..7
        - 若 x 是 int 且在 [-8..7]：直接返回（认为已是十进制有符号值）
        - 若 x 是 0..15 或二进制字符串：先当作 bit pattern，再转 two's complement
        """
        if isinstance(x, (int, np.integer)):
            v = int(x)
            if -8 <= v <= 7:
                return v
            if 0 <= v <= 15:
                return v - 16 if v >= 8 else v
            raise ValueError(f"A 元素十进制值超出可支持范围（-8..7 或 0..15）: {v}")

        u = _parse_bin_str_4(x)  # 0..15
        return u - 16 if u >= 8 else u

    def _bin4_to_uint(x) -> int:
        """4bit 无符号：0..15"""
        return _parse_bin_str_4(x)

    def _bin4_to_sint(x) -> int:
        """4bit 有符号 two's complement：-8..7"""
        return _parse_sint_4(x)

    # =========================
    # ✅ 修改：_pre_transform_A 依据 pe_mode 选择有符号/无符号解码
    #    新规则：gemm 有符号；cnn/snn 无符号
    # =========================
    def _pre_transform_A(A, pe_mode_in_norm_int: int, step_mode_in_int: int):
        A_arr = np.array(A)
        flat = A_arr.reshape(-1)

        # gemm：有符号
        if pe_mode_in_norm_int == 2:
            flat_int = np.array([_bin4_to_sint(v) for v in flat], dtype=int)
            return flat_int.reshape(A_arr.shape).astype(int)

        # cnn / snn：无符号
        flat_int = np.array([_bin4_to_uint(v) for v in flat], dtype=int)

        # cnn：不做映射
        if pe_mode_in_norm_int == 0:
            return flat_int.reshape(A_arr.shape).astype(int)

        # snn：做映射
        step = int(step_mode_in_int)
        if step == 4:
            mp = _SNN_MAP_4
        elif step == 8:
            mp = _SNN_MAP_8
        elif step == 12:
            mp = _SNN_MAP_12
        else:
            raise ValueError(f"snn 只支持 step_mode=4/8/12，但收到 {step}")

        mapped = np.empty_like(flat_int)
        for i, val in enumerate(flat_int):
            if val not in mp:
                raise ValueError(f"snn step={step} 时，A 元素值 {val} 不在映射表内")
            mapped[i] = mp[val]

        return mapped.reshape(A_arr.shape).astype(int)

    # ========== txt 保存 ==========
    def _rows_to_32bit_concat(arr_2d):
        arr = np.array(arr_2d)
        if arr.ndim != 2:
            return np.array(arr, dtype=str).reshape(-1)

        N, C = arr.shape
        if C != 8:
            return np.array(["".join(map(str, row)) for row in arr], dtype=str)

        out = []
        for r in range(N):
            bits = []
            for v in arr[r, :]:
                s = str(v)
                if all(ch in "01" for ch in s) and len(s) <= 4:
                    bits.append(s.zfill(4))
                else:
                    bits.append(format(int(s), "04b"))
            out.append("".join(bits))
        return np.array(out, dtype=str)

    # ✅ 修复版：按 group_cnt 生成 a_{idx}_{1..group_cnt}.txt
    def _split_and_save_a(a, idx: int, depth: int, mode_txt: int, group_cnt: int):
        base_prefix = f"a_{idx}"

        if a is None:
            with open(base_prefix + ".txt", "w", encoding="utf-8") as f:
                f.write("None\n")
            return

        if depth <= 0:
            raise ValueError(f"内部错误：depth={depth} 非法")

        if not isinstance(group_cnt, (int, np.integer)):
            raise TypeError(f"group_cnt 必须是 int，但收到 {type(group_cnt)}: {group_cnt}")
        group_cnt = int(group_cnt)
        if group_cnt <= 0:
            arr = np.array(a)
            if arr.ndim == 1 and arr.dtype.kind in ("U", "S", "O"):
                lines = arr.astype(str).reshape(-1)
            else:
                lines = _rows_to_32bit_concat(arr).reshape(-1)
            np.savetxt(f"{base_prefix}_1.txt", lines, fmt="%s")
            return

        arr = np.array(a)
        if arr.ndim == 1 and arr.dtype.kind in ("U", "S", "O"):
            lines = arr.astype(str).reshape(-1)
        else:
            lines = _rows_to_32bit_concat(arr).reshape(-1)

        Q = len(lines)
        if Q == 0:
            with open(base_prefix + "_empty.txt", "w", encoding="utf-8"):
                pass
            return

        start_idx = 0
        # 保留你原来的 GEMM head2 规则
        if mode_txt == 5 and Q >= 2:
            np.savetxt(base_prefix + "_head2.txt", lines[:2], fmt="%s")
            start_idx = 2

        cur = start_idx
        for g in range(1, group_cnt + 1):
            if g < group_cnt:
                end = min(cur + depth, Q)
            else:
                end = Q

            chunk = lines[cur:end]
            np.savetxt(f"{base_prefix}_{g}.txt", chunk, fmt="%s")
            cur = end

            if cur >= Q:
                for gg in range(g + 1, group_cnt + 1):
                    with open(f"{base_prefix}_{gg}.txt", "w", encoding="utf-8"):
                        pass
                break

    def _save_w_to_txt_32bit(w, filename: str):
        if w is None:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("None\n")
            return

        arr = np.array(w)
        if arr.ndim == 2 and arr.shape[1] == 8:
            out_lines = _rows_to_32bit_concat(arr)
        elif arr.ndim == 1 and arr.dtype.kind in ("U", "S", "O"):
            out_lines = arr.astype(str)
        else:
            if arr.ndim == 2:
                out_lines = np.array(["".join(map(str, row)) for row in arr], dtype=str)
            else:
                out_lines = arr.reshape(-1).astype(str)

        np.savetxt(filename, out_lines, fmt="%s")

    def _fix_kernel_size_pe_step(res_tuple, kernel_size_in_std: str):
        (a, w, depth, pe_mode_out, step_mode_out, _k_old, _enable_out, a_out) = res_tuple
        return (
            a,
            w,
            depth,
            _map_pe_mode_out(pe_mode_out),
            _normalize_step_mode_out_str(step_mode_out),
            _normalize_kernel_size_str(kernel_size_in_std),
            a_out,   # group / a_out
        )

    def _to_dec_string(x, name: str) -> str:
        if isinstance(x, str):
            s = x.strip()
            if s.lstrip("-").isdigit():
                return s
            raise ValueError(f"{name} 期望是十进制数字/数字字符串，但收到: {repr(x)}")
        try:
            return str(int(x))
        except Exception as e:
            raise ValueError(f"{name} 期望可转为 int 的数值，但收到 {type(x)}: {x}") from e

    def _to_int(x, name: str) -> int:
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, str):
            s = x.strip()
            if s.lstrip("-").isdigit():
                return int(s)
            raise ValueError(f"{name} 期望是数字/数字字符串，但收到: {repr(x)}")
        try:
            return int(x)
        except Exception as e:
            raise ValueError(f"{name} 期望可转为 int 的数值，但收到 {type(x)}: {x}") from e

    # =========================
    # 逐路调用 transform_all（由 kernel_num 决定）
    # =========================
    outs = []
    mode_list_int = []

    for idx in range(kernel_num):
        A  = A_list[idx]
        A2 = A2_list[idx]
        W  = W_list[idx]

        pe_mode_in_raw = pe_mode_in_list[idx]
        pe_mode_in = _normalize_pe_mode_in(pe_mode_in_raw)  # -> 0/1/2

        _dial_str_std, dial_call = _normalize_dilation_str(dialation_in_list[idx])
        kernel_size_in_std = _normalize_kernel_size_str(kernel_size_in_list[idx])

        step_mode_in_str = _normalize_step_mode_str(step_mode_in_list[idx])  # "4"/"8"/"12"
        step_mode_in_int = int(step_mode_in_str)

        scale = scale_list[idx]

        enable_in = 1
        to_binary = True
        bit_width = 4

        mode_txt = _compute_mode(pe_mode_in, dial_call, kernel_size_in_std)
        mode_list_int.append(mode_txt)

        A_pre_int = _pre_transform_A(A, pe_mode_in, step_mode_in_int)

        # 保持原设计：transform_all 由你工程提供
        out_raw = transform_all(
            A_pre_int, A2, W, mode_txt,
            pe_mode_in, step_mode_in_int, enable_in,
            to_binary, bit_width,
            scale=scale
        )

        outs.append(_fix_kernel_size_pe_step(out_raw, kernel_size_in_std))

    # =========================
    # 写 txt + 汇总
    # =========================
    depth_list_int = []
    pe_mode_list = []
    step_mode_list = []
    kernel_size_list = []
    group_list_int_raw = []

    for idx, (res, mode_txt) in enumerate(zip(outs, mode_list_int)):
        (a, w, depth, pe_mode, step_mode_str, kernel_size, group) = res

        _split_and_save_a(a=a, idx=idx, depth=depth, mode_txt=mode_txt, group_cnt=group)
        _save_w_to_txt_32bit(w, f"w_{idx}.txt")

        depth_list_int.append(depth)
        pe_mode_list.append(pe_mode)
        step_mode_list.append(step_mode_str)
        kernel_size_list.append(kernel_size)
        group_list_int_raw.append(group)

    mode_list = [_to_dec_string(v, "mode_list") for v in mode_list_int]
    depth_list = [_to_int(v, "depth_list") for v in depth_list_int]
    group_list = [_to_int(v, "group_list") for v in group_list_int_raw]
    group_max = max(group_list) if len(group_list) > 0 else None

    return tuple(outs) + (
        kernel_num,          # int 原样输出
        mode_list,           # list[str]
        depth_list,          # list[int]
        pe_mode_list,        # list[str]
        step_mode_list,      # list[str]
        kernel_size_list,    # list[str]
        group_list,          # list[int]
        group_max,           # int
    )




#=====================================================================================================


from typing import List
import secrets


def Decompile_3_3(data_in: List[List[str]], depth: int) -> List[List[str]]:
    """
    对输入矩阵 data_in 做分组变换：
      - 每 (depth-2) 行为一个大组
      - 每个大组：先转置，再把转置后的矩阵做“首行↔末行、次行↔倒二行...”的对调
        （等价于：转置矩阵的行整体反转）
      - 所有大组的结果按行顺序拼接，得到输出矩阵

    说明：
      - 输入每个元素是 8-bit 字符串（但函数本身只做位置变换）
      - 若每组原尺寸是 (depth-2) x C，则输出每组尺寸是 C x (depth-2)
    """
    if depth <= 2:
        raise ValueError("depth 必须 > 2（保证 depth-2 为正）")

    group_rows = depth - 2
    if not data_in:
        return []

    # 检查矩阵列数一致
    col_len = len(data_in[0])
    if col_len == 0:
        raise ValueError("data_in 列数不能为 0（存在空行）")
    for i, row in enumerate(data_in):
        if len(row) != col_len:
            raise ValueError(f"data_in 第 {i} 行列数不一致：期望 {col_len}，实际 {len(row)}")

    # 检查总行数能整除 group_rows
    total_rows = len(data_in)
    if total_rows % group_rows != 0:
        raise ValueError(
            f"data_in 总行数 {total_rows} 不能被 (depth-2)={group_rows} 整除，无法完整分组"
        )

    # 分组处理
    n_groups = total_rows // group_rows
    out: List[List[str]] = []

    for g in range(n_groups):
        block = data_in[g * group_rows: (g + 1) * group_rows]  # group_rows x col_len

        # 1) 转置：col_len x group_rows
        transposed = [list(col) for col in zip(*block)]

        # 2) 首尾行对调（等价于行反转）
        transformed = transposed[::-1]

        # 3) 拼接输出
        out.extend(transformed)

    return out



#======================================================================================


def Decompile_1_3(data_in: List[List[str]], depth: int) -> List[List[str]]:
    """
    对输入矩阵 data_in 做分组矩阵变换：
      - 从第 1 行开始，每 (depth-2) 行为一个“大组”
      - 对每个大组：先转置，然后把转置矩阵做首行↔末行、次行↔倒二行...（等价：行反转）
      - 各大组结果按行顺序依次拼接，形成输出矩阵

    说明：
      - 输入元素为 8-bit 字符串，但本函数仅做位置变换
      - 若每组原尺寸为 (depth-2) x C，则变换后每组尺寸为 C x (depth-2)
    """
    if depth <= 2:
        raise ValueError("depth 必须 > 2（保证 depth-2 为正）")

    group_rows = depth - 2
    if not data_in:
        return []

    # 检查列数一致
    col_len = len(data_in[0])
    if col_len == 0:
        raise ValueError("data_in 列数不能为 0（存在空行）")
    for i, row in enumerate(data_in):
        if len(row) != col_len:
            raise ValueError(f"data_in 第 {i} 行列数不一致：期望 {col_len}，实际 {len(row)}")

    # 检查总行数能整除 group_rows
    total_rows = len(data_in)
    if total_rows % group_rows != 0:
        raise ValueError(
            f"data_in 总行数 {total_rows} 不能被 (depth-2)={group_rows} 整除，无法完整分组"
        )

    n_groups = total_rows // group_rows
    out: List[List[str]] = []

    for g in range(n_groups):
        block = data_in[g * group_rows: (g + 1) * group_rows]  # group_rows x col_len

        # 1) 转置：col_len x group_rows
        transposed = [list(col) for col in zip(*block)]

        # 2) 转置后首尾行对调（等价于行反转）
        transformed = transposed[::-1]

        # 3) 拼接输出
        out.extend(transformed)

    return out



#======================================================================================


from typing import List


def Decompile_3_1(data_in: List[List[str]], depth: int) -> List[List[str]]:
    """
    Decompile_3_1：
      1) 每 (depth-2) 行为一个大组
      2) 每组：转置 -> 转置矩阵行反转（首行<->末行...）
      3) 组结果按行拼接
    """
    if depth <= 2:
        raise ValueError("depth 必须 > 2（保证 depth-2 为正）")

    group_rows = depth - 2
    if not data_in:
        return []

    col_len = len(data_in[0])
    if col_len == 0:
        raise ValueError("data_in 列数不能为 0（存在空行）")
    for i, row in enumerate(data_in):
        if len(row) != col_len:
            raise ValueError(f"data_in 第 {i} 行列数不一致：期望 {col_len}，实际 {len(row)}")

    total_rows = len(data_in)
    if total_rows % group_rows != 0:
        raise ValueError(
            f"data_in 总行数 {total_rows} 不能被 (depth-2)={group_rows} 整除，无法完整分组"
        )

    n_groups = total_rows // group_rows

    stitched: List[List[str]] = []
    for g in range(n_groups):
        block = data_in[g * group_rows: (g + 1) * group_rows]  # (depth-2) x C
        transposed = [list(col) for col in zip(*block)]        # C x (depth-2)
        transformed = transposed[::-1]                         # C x (depth-2)  (行反转)
        stitched.extend(transformed)                           # (groups*C) x (depth-2)

    return stitched



#======================================================================================



from typing import List, Any


def Decompile_dialted_2_1(data_in: List[List[str]], depth: int) -> List[List[str]]:

    # ----------------
    # 基本检查
    # ----------------
    if not isinstance(depth, int):
        raise TypeError("depth 必须是 int")
    if depth <= 0:
        raise ValueError("depth 必须为正整数")
    if depth < 2:
        raise ValueError("depth 过小：至少需要 2")
    if depth % 2 == 1 and depth < 3:
        raise ValueError("depth 为奇数时至少需要 3（否则下半每组删掉最后一行后会变成空组）")

    if not data_in:
        return []

    Y = len(data_in)
    if Y <= 0:
        raise ValueError(f"输入矩阵行数 Y 必须为正，当前 Y={Y}")

    X = len(data_in[0])
    if X <= 0:
        raise ValueError("输入矩阵列数 X 不能为 0")

    # 检查每行列数一致
    for i, row in enumerate(data_in):
        if len(row) != X:
            raise ValueError(f"第 {i} 行列数不一致：期望 {X}，实际 {len(row)}")

    # ----------------
    # 切 top / bot
    # ----------------
    if Y % 2 != 0:
        raise ValueError(f"要求 Y 为偶数才能平分上下半矩阵，但当前 Y={Y}")

    half = Y // 2
    if half % depth != 0:
        raise ValueError(f"(Y/2)={half} 必须能被 depth={depth} 整除，才能形成整数个大组")

    groups = half // depth
    top = data_in[:half]
    bot = data_in[half:]

    # ----------------
    # 组内变换：transpose + reverse rows（保持与原 txt 一致）:contentReference[oaicite:1]{index=1}
    # ----------------
    def transpose_and_reverse_rows(block: List[List[str]]) -> List[List[str]]:
        # block: R x X -> zip(*block): X x R，再 [::-1] 反转“行”（即转置矩阵的行序）
        return [list(col) for col in zip(*block)][::-1]

    stitched: List[List[str]] = []

    Rt = depth
    for g in range(groups):
        top_block = top[g * depth:(g + 1) * depth]  # depth x X
        bot_block = bot[g * depth:(g + 1) * depth]  # depth x X

        # depth 为奇数：下半每个大组删掉最后一行
        if depth % 2 == 1:
            bot_block = bot_block[:-1]  # (depth-1) x X

        Rb = len(bot_block)  # depth(偶) or depth-1(奇)

        top_trans = transpose_and_reverse_rows(top_block)  # X x Rt
        bot_trans = transpose_and_reverse_rows(bot_block)  # X x Rb（奇数时更短）

        # 合并方式：按列交错（保持与原 txt 一致）:contentReference[oaicite:2]{index=2}
        out_block: List[List[str]] = []
        max_r = max(Rt, Rb)
        for i in range(X):
            row_out: List[str] = []
            for j in range(max_r):
                if j < Rt:
                    row_out.append(top_trans[i][j])
                if j < Rb:
                    row_out.append(bot_trans[i][j])
            out_block.append(row_out)

        stitched.extend(out_block)

    return stitched





#======================================================================================


from typing import List, Any


def Decompile_dialted_1_2(data_in: List[List[str]], depth: int) -> List[List[str]]:

    # ----------------
    # 基本检查
    # ----------------
    if not isinstance(depth, int):
        raise TypeError("depth 必须是 int")
    if depth <= 0:
        raise ValueError("depth 必须为正整数")
    if depth < 2:
        raise ValueError("depth 过小：至少需要 2")
    if depth % 2 == 1 and depth < 3:
        raise ValueError("depth 为奇数时至少需要 3（否则下半每组删掉最后一行后会变成空组）")

    if not data_in:
        return []

    Y = len(data_in)
    if Y <= 0:
        raise ValueError(f"输入矩阵行数 Y 必须为正，当前 Y={Y}")

    X = len(data_in[0])
    if X <= 0:
        raise ValueError("输入矩阵列数 X 不能为 0")

    # 检查每行列数一致
    for i, row in enumerate(data_in):
        if len(row) != X:
            raise ValueError(f"第 {i} 行列数不一致：期望 {X}，实际 {len(row)}")

    # ----------------
    # 切 top / bot
    # ----------------
    if Y % 2 != 0:
        raise ValueError(f"要求 Y 为偶数才能平分上下半矩阵，但当前 Y={Y}")

    half = Y // 2
    if half % depth != 0:
        raise ValueError(f"(Y/2)={half} 必须能被 depth={depth} 整除，才能形成整数个大组")

    groups = half // depth
    top = data_in[:half]
    bot = data_in[half:]

    # ----------------
    # 组内变换：transpose + reverse rows（保持与原 txt 一致）:contentReference[oaicite:1]{index=1}
    # ----------------
    def transpose_and_reverse_rows(block: List[List[str]]) -> List[List[str]]:
        # block: R x X -> zip(*block): X x R，再 [::-1] 反转“行”（即转置矩阵的行序）
        return [list(col) for col in zip(*block)][::-1]

    stitched: List[List[str]] = []

    Rt = depth
    for g in range(groups):
        top_block = top[g * depth:(g + 1) * depth]  # depth x X
        bot_block = bot[g * depth:(g + 1) * depth]  # depth x X

        # depth 为奇数：下半每个大组删掉最后一行
        if depth % 2 == 1:
            bot_block = bot_block[:-1]  # (depth-1) x X

        Rb = len(bot_block)  # depth(偶) or depth-1(奇)

        top_trans = transpose_and_reverse_rows(top_block)  # X x Rt
        bot_trans = transpose_and_reverse_rows(bot_block)  # X x Rb（奇数时更短）

        # 合并方式：按列交错（保持与原 txt 一致）:contentReference[oaicite:2]{index=2}
        out_block: List[List[str]] = []
        max_r = max(Rt, Rb)
        for i in range(X):
            row_out: List[str] = []
            for j in range(max_r):
                if j < Rt:
                    row_out.append(top_trans[i][j])
                if j < Rb:
                    row_out.append(bot_trans[i][j])
            out_block.append(row_out)

        stitched.extend(out_block)

    return stitched



#======================================================================================



def Decompile_gemm(data_in: List[List[str]], a: int) -> List[List[str]]:
    """
    Decompile_gemm:
      1) 输入矩阵 data_in 为 Y x X
      2) 每行 X 个元素按顺序拼接成 1 个“大元素”（字符串拼接），得到 Y x 1
      3) 以 a 为分组大小：从上到下每 a 行重排列到输出矩阵的一行，从左到右依次放置
         例如原来 (a,1) 对应的“行拼接后元素”会放到输出矩阵的 (1,a)

    约束（按你的说明）：
      - Y 一定能被 a 整除；若不整除，直接报错
    """
    if a <= 0:
        raise ValueError("a 必须为正整数")
    if not data_in:
        return []

    Y = len(data_in)
    X = len(data_in[0])
    if X == 0:
        raise ValueError("输入矩阵列数 X 不能为 0")

    # 检查列数一致
    for i, row in enumerate(data_in):
        if len(row) != X:
            raise ValueError(f"第 {i} 行列数不一致：期望 {X}，实际 {len(row)}")

    # 强约束：Y 必须能被 a 整除
    if Y % a != 0:
        raise ValueError(f"输入矩阵行数 Y={Y} 必须能被 a={a} 整除（按设计约束）")

    # 1) 每行拼接成一个“大元素”：宽度扩大 X 倍（若每元素等宽）
    merged_rows: List[str] = ["".join(row) for row in data_in]  # 长度 = sum(len(cell))

    # 2) 每 a 行重排成输出矩阵的一行
    out: List[List[str]] = []
    for start in range(0, Y, a):
        out.append(merged_rows[start:start + a])  # 这一行一定正好长度 a

    return out


#======================================================================================


from typing import List
import secrets


def Decompile_3_3_2s(data_in: List[List[str]], depth: int) -> List[List[str]]:
 
    if depth <= 2:
        raise ValueError("depth 必须 > 2（保证 depth-2 为正）")
    R = depth - 2

    if not data_in:
        return []

    # 检查列数一致
    C = len(data_in[0])
    if C == 0:
        raise ValueError("data_in 列数不能为 0")
    for i, row in enumerate(data_in):
        if len(row) != C:
            raise ValueError(f"data_in 第 {i} 行列数不一致：期望 {C}，实际 {len(row)}")

    # 行数必须可按 R 分组
    Y = len(data_in)
    if Y % R != 0:
        raise ValueError(f"data_in 总行数 {Y} 必须能被 (depth-2)={R} 整除，才能完整分组")

    # 1) 删除奇数列后：保留 0-based: 1,3,5,...
    kept_col_indices = list(range(1, C, 2))
    if len(kept_col_indices) == 0:
        raise ValueError(f"删除奇数列后没有剩余列（原始列数 C={C} 太小）")

    out: List[List[str]] = []
    n_groups = Y // R

    for g in range(n_groups):
        block = data_in[g * R:(g + 1) * R]  # R x C

        # 1) 删除奇数列（保留偶数列）
        block_kept = [[row[j] for j in kept_col_indices] for row in block]  # R x C2

        # 2) 转置：C2 x R
        transposed = [list(col) for col in zip(*block_kept)]  # C2 x R

        # 3) 行反转（首尾对调...）
        transformed = transposed[::-1]  # C2 x R

        # 4) ★新增：删除每组中的偶数列（1-based第2,4,...列删 => 保留0-based偶数索引0,2,4,...）
        transformed_filtered = [row[0::2] for row in transformed]

        # 5) 拼接
        out.extend(transformed_filtered)

    return out



#===========================================================================================



from typing import List, Tuple
import random


def _transpose(mat: List[List[int]]) -> List[List[int]]:
    return [list(col) for col in zip(*mat)]


def _group_transform(block: List[List[int]]) -> List[List[int]]:
    """
    对一个大组 block (R x C) 做：
      1) 转置 -> (C x R)
      2) 转置后矩阵做首行<->末行、次行<->倒二行...（等价：行反转）
    返回 transformed: (C x R)
    """
    transposed = _transpose(block)       # C x R
    transformed = transposed[::-1]       # 行反转
    return transformed


def Decompile_6_6(data_in: List[List[int]], depth: int) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Decompile_6_6:
      1) data_in 行数为 4a（a为正整数）
      2) 从上到下按行分成 B1,B2,B3,B4，每个 a 行，列数相同
      3) C = B1 + B2 + B3 + B4（逐元素相加）
      4) 对 C（行数为 a）从第1行开始，每 (depth-2) 行为一组：
           - 每组做：转置 + 转置后行反转（首尾对调...）
         各组结果按行拼接，形成 out
    返回:
      C, out
    """
    if depth <= 2:
        raise ValueError("depth 必须 > 2（保证 depth-2 为正）")
    R = depth - 2

    if not data_in:
        raise ValueError("data_in 不能为空")

    total_rows = len(data_in)
    if total_rows % 4 != 0:
        raise ValueError(f"data_in 行数必须是 4a，当前行数={total_rows}")

    a = total_rows // 4
    if a <= 0:
        raise ValueError("a 必须为正整数")

    cols = len(data_in[0])
    if cols == 0:
        raise ValueError("data_in 列数不能为 0")

    # 检查每行列数一致
    for i, row in enumerate(data_in):
        if len(row) != cols:
            raise ValueError(f"第 {i} 行列数不一致：期望 {cols}，实际 {len(row)}")

    # 切分 B1..B4
    B1 = data_in[0:a]
    B2 = data_in[a:2*a]
    B3 = data_in[2*a:3*a]
    B4 = data_in[3*a:4*a]

    # 逐元素相加得到 C (a x cols)
    C: List[List[int]] = []
    for r in range(a):
        C.append([B1[r][c] + B2[r][c] + B3[r][c] + B4[r][c] for c in range(cols)])

    # C 的行数 a 必须能按 (depth-2) 分组
    if a % R != 0:
        raise ValueError(f"C 的行数 a={a} 必须能被 (depth-2)={R} 整除，才能完整分组")

    n_groups = a // R
    out: List[List[int]] = []

    # 分组变换并拼接
    for g in range(n_groups):
        block = C[g * R:(g + 1) * R]      # R x cols
        transformed = _group_transform(block)  # cols x R
        out.extend(transformed)

    return C, out



#=============================================================================



from typing import List, Any, Optional


# ==============================================================================
# 新增子函数：Decompile_3_3_2s（你刚刚要的版本）
#   规则：每 (depth-2) 行为一组：
#       1) 删除奇数列（1-based: 1,3,5...；即 0-based: 0,2,4...）
#       2) 转置
#       3) 转置后行翻转（首行<->末行，等价于 reverse rows）
# ==============================================================================
def Decompile_3_3_2s(data_in: List[List[str]], depth: int) -> List[List[str]]:
 
    if depth <= 2:
        raise ValueError("depth 必须 > 2（保证 depth-2 为正）")
    R = depth - 2

    if not data_in:
        return []

    # 检查列数一致
    C = len(data_in[0])
    if C == 0:
        raise ValueError("data_in 列数不能为 0")
    for i, row in enumerate(data_in):
        if len(row) != C:
            raise ValueError(f"data_in 第 {i} 行列数不一致：期望 {C}，实际 {len(row)}")

    # 行数必须可按 R 分组
    Y = len(data_in)
    if Y % R != 0:
        raise ValueError(f"data_in 总行数 {Y} 必须能被 (depth-2)={R} 整除，才能完整分组")

    # 1) 删除奇数列后：保留 0-based: 1,3,5,...
    kept_col_indices = list(range(1, C, 2))
    if len(kept_col_indices) == 0:
        raise ValueError(f"删除奇数列后没有剩余列（原始列数 C={C} 太小）")

    out: List[List[str]] = []
    n_groups = Y // R

    for g in range(n_groups):
        block = data_in[g * R:(g + 1) * R]  # R x C

        # 1) 删除奇数列（保留偶数列）
        block_kept = [[row[j] for j in kept_col_indices] for row in block]  # R x C2

        # 2) 转置：C2 x R
        transposed = [list(col) for col in zip(*block_kept)]  # C2 x R

        # 3) 行反转（首尾对调...）
        transformed = transposed[::-1]  # C2 x R

        # 4) ★新增：删除每组中的偶数列（1-based第2,4,...列删 => 保留0-based偶数索引0,2,4,...）
        transformed_filtered = [row[0::2] for row in transformed]

        # 5) 拼接
        out.extend(transformed_filtered)

    return out




from typing import List, Optional, Any
import math

# ------------------------------------------------------------
# rows_num：按你给的规则计算最终应保留的行数 H_out
# ------------------------------------------------------------
def rows_num(mode_all: int, height: int, stride: int) -> int:
    H = int(height)
    S = int(stride)
    if S <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    if mode_all == 5:
        return H
    if mode_all == 6:
        return H - 3

    if mode_all in (0, 1, 2, 3, 4):
        K = 3
        P = 0
        D = 2 if mode_all == 4 else 1
        # floor( (H + 2P - D*(K-1) - 1)/S + 1 )
        return math.floor((H + 2 * P - D * (K - 1) - 1) / S + 1)

    raise ValueError(f"Unsupported mode_all={mode_all}, expected 0~6.")



#=====================================================================================================================================================================================================================================

# ==============================================================================
#  FULL: Data_rearrange + Split64_to_8cols + Data_discarded + Decompile_all
#  TESTBENCH INCLUDED
#
#  IMPORTANT:
#    你需要确保以下 Decompile_xxx 函数已在当前文件中定义或已 import：
#      Decompile_3_3, Decompile_3_3_2s, Decompile_1_3, Decompile_3_1,
#      Decompile_dialted_1_2, Decompile_dialted_2_1, Decompile_gemm, Decompile_6_6
# ==============================================================================

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import secrets


# ==============================================================================
# CoreRecord + normalize（关键修改：兼容 'mode' 和 'pe_mode'）
# ==============================================================================

@dataclass
class CoreRecord:
    group_id: int
    pe_mode: str
    step_mode: int
    mode_all: int
    depth: int
    data: List[str]


def make_group_random_64bit_lines(group_id: int, n: int) -> List[str]:
    out: List[str] = []
    for i in range(n):
        hi48 = secrets.randbits(48)
        lo16 = ((group_id & 0xFF) << 8) | (i & 0xFF)
        val = (hi48 << 16) | lo16
        out.append(format(val, "064b"))
    return out


def _unwrap_scalar_maybe_list(v: Any, name: str) -> Any:
    if isinstance(v, (list, tuple)):
        if len(v) == 1:
            return v[0]
        raise ValueError(f"{name} 不应是多元素 list/tuple：{v}")
    return v


def _dict_to_corerecord(d: Dict[str, Any]) -> CoreRecord:
    pe_mode_raw = d.get("pe_mode", None)
    if pe_mode_raw is None:
        pe_mode_raw = d.get("mode", None)

    group_id  = int(_unwrap_scalar_maybe_list(d.get("group_id"), "group_id"))
    pe_mode   = str(_unwrap_scalar_maybe_list(pe_mode_raw, "pe_mode/mode"))
    step_mode = int(_unwrap_scalar_maybe_list(d.get("step_mode"), "step_mode"))
    mode_all  = int(_unwrap_scalar_maybe_list(d.get("mode_all"), "mode_all"))
    depth     = int(_unwrap_scalar_maybe_list(d.get("depth"), "depth"))

    data = d.get("data")
    if not isinstance(data, (list, tuple)):
        raise ValueError(f"data 必须是 List[str]，当前={type(data)}")
    data = list(data)
    if not all(isinstance(x, str) for x in data):
        raise ValueError("data 中必须全部是 str（每行 64-bit）")

    return CoreRecord(group_id, pe_mode, step_mode, mode_all, depth, data)


def _normalize_records(records: List[Any]) -> List[CoreRecord]:
    if not records:
        return []
    if isinstance(records[0], CoreRecord):
        return records  # type: ignore[return-value]
    if isinstance(records[0], dict):
        return [_dict_to_corerecord(r) for r in records]  # type: ignore[arg-type]
    raise TypeError(f"records 的元素类型不支持：{type(records[0])}")


# ==============================================================================
# Data_rearrange / Split64_to_8cols / Data_discarded / Decompile_all（debug 版）
# ==============================================================================

def Data_rearrange(records: List[CoreRecord]) -> List[List[str]]:
    records_sorted = sorted(records, key=lambda r: r.group_id)

    grouped: Dict[int, List[str]] = {}
    max_gid = -1
    for r in records_sorted:
        max_gid = max(max_gid, r.group_id)
        grouped.setdefault(r.group_id, []).extend(r.data)

    merged_rows: List[str] = []
    for gid in range(max_gid + 1):
        merged_rows.extend(grouped.get(gid, []))

    return [[row] for row in merged_rows]


def Split64_to_8cols(big_matrix_64: List[List[str]]) -> List[List[str]]:
    matrix_8cols: List[List[str]] = []
    for idx, row in enumerate(big_matrix_64):
        s = row[0]
        if len(s) != 64 or (set(s) - {"0", "1"}):
            raise ValueError(f"big_matrix_64 第 {idx} 行不是 64-bit：{s}")
        matrix_8cols.append([s[i:i + 8] for i in range(0, 64, 8)])
    return matrix_8cols


def _discard_bits_rule(pe_mode: str, step_mode: int, mode_all: int) -> int:
    if mode_all in (0, 6):
        if pe_mode == "cnn":
            return 16
        if pe_mode == "snn" and step_mode in (4, 8):
            return 16
        if pe_mode == "snn" and step_mode == 12:
            return 32
        return 0

    if mode_all in (1, 2, 3, 4):
        if pe_mode == "cnn":
            return 0
        if pe_mode == "snn" and step_mode in (4, 8):
            return 0
        if pe_mode == "snn" and step_mode == 12:
            return 16
        return 0

    if mode_all == 5:
        return 32

    return 0


def Data_discarded(matrix_8cols: List[List[str]], record) -> Tuple[List[List[str]], int, int]:
    discard_bits = _discard_bits_rule(record.pe_mode, record.step_mode, record.mode_all)
    discard_bytes = discard_bits // 8

    kept_bits = 64 - discard_bits
    kept_cols = kept_bits // 8  

    new_matrix: List[List[str]] = []
    for row in matrix_8cols:
        if record.mode_all == 5 and discard_bytes > 0:
            new_matrix.append(row[:-discard_bytes])
        else:
            new_matrix.append(row[discard_bytes:])

    return new_matrix, discard_bits, kept_bits


# ------------------- 打印工具（核心：方便核对中间矩阵） -------------------

def _shape_of(x: Any) -> str:
    if isinstance(x, tuple) and len(x) == 2:
        return f"(tuple) C={_shape_of(x[0])}, out={_shape_of(x[1])}"
    if isinstance(x, list):
        if not x:
            return "(0,0)"
        if isinstance(x[0], list):
            return f"({len(x)},{len(x[0])})"
        return f"({len(x)},)"
    return str(type(x))


def print_matrix(mat: Any, title: str, max_rows: Optional[int] = 20, max_cols: Optional[int] = 16) -> None:
    print("\n" + "-" * 110)
    print(f"{title}  shape={_shape_of(mat)}")

    def _print_list2(m: List[List[Any]]):
        rows = m if max_rows is None else m[:max_rows]
        for i, row in enumerate(rows):
            if isinstance(row, list) and max_cols is not None and len(row) > max_cols:
                show = row[:max_cols]
                print(f"{i:04d}: {show} ... (+{len(row)-max_cols} cols)")
            else:
                print(f"{i:04d}: {row}")
        if max_rows is not None and len(m) > max_rows:
            print(f"... (+{len(m)-max_rows} rows)")

    if isinstance(mat, tuple) and len(mat) == 2:
        C, out = mat
        print_matrix(C, title + " (C)", max_rows=max_rows, max_cols=max_cols)
        print_matrix(out, title + " (out)", max_rows=max_rows, max_cols=max_cols)
        return

    if isinstance(mat, list) and (len(mat) == 0 or isinstance(mat[0], list)):
        _print_list2(mat)  # type: ignore[arg-type]
        return

    if isinstance(mat, list):
        rows = mat if max_rows is None else mat[:max_rows]
        for i, v in enumerate(rows):
            print(f"{i:04d}: {v}")
        if max_rows is not None and len(mat) > max_rows:
            print(f"... (+{len(mat)-max_rows} rows)")
        return

    print(mat)


# ------------------- Debug 版 Decompile_all：打印每一步中间结果 -------------------

from typing import Any, List, Optional


def Decompile_all_debug(
    records: List[Any],
    stride: int,
    depadding_num: int,
    padding_one_dilated: int,
    *,
    depth: int,
    core_num: Optional[int] = None,
    max_rows: Optional[int] = 20,
    max_cols: Optional[int] = 16,
    a1_height: int,
) -> Any:
    records = _normalize_records(records)
    if not records:
        raise ValueError("records 不能为空")

    if not isinstance(depth, int):
        raise TypeError("depth 必须是 int")
    if depth <= 0:
        raise ValueError("depth 必须为正整数")

    if not isinstance(padding_one_dilated, int):
        raise TypeError("padding_one_dilated 必须是 int")
    if padding_one_dilated not in (0, 1):
        raise ValueError("padding_one_dilated 只能为 0 或 1")

    ref = records[0]
    for i, r in enumerate(records[1:], start=1):
        if (r.pe_mode, r.step_mode, r.mode_all, r.depth) != (ref.pe_mode, ref.step_mode, ref.mode_all, ref.depth):
            raise ValueError(f"records 参数不一致：idx={i}")

    head = f"[Decompile_all_debug] core_num={core_num}" if core_num is not None else "[Decompile_all_debug]"
    print("\n" + "=" * 120)
    print(head)

    # 同时打印 record 自带的 depth 与 dialted 使用的 depth，便于核对
    print(
        f"meta: pe_mode={ref.pe_mode}, mode_all={ref.mode_all}, "
        f"record_depth={ref.depth}, dialted_depth={depth}, "
        f"step_mode={ref.step_mode}, stride={stride}, depadding_num={depadding_num}, "
        f"padding_one_dilated={padding_one_dilated}"
    )
    print(f"num_group_records={len(records)}, group_ids={[r.group_id for r in records]}")

    # Stage1: Data_rearrange
    big_matrix_64 = Data_rearrange(records)
    print_matrix(
        big_matrix_64,
        "Stage1: Data_rearrange output (64-bit rows in 1-col matrix)",
        max_rows=max_rows,
        max_cols=max_cols,
    )

    # Stage2: Split64_to_8cols
    matrix_8cols = Split64_to_8cols(big_matrix_64)
    print_matrix(
        matrix_8cols,
        "Stage2: Split64_to_8cols output (8 cols of 8-bit strings)",
        max_rows=max_rows,
        max_cols=max_cols,
    )

    # Stage3: Data_discarded
    data_in, discard_bits, kept_bits = Data_discarded(matrix_8cols, ref)
    print(f"\nStage3: Data_discarded rule => discard_bits={discard_bits}, kept_bits={kept_bits}")
    print_matrix(
        data_in,
        "Stage3: Data_discarded output (after dropping MSB bytes)",
        max_rows=max_rows,
        max_cols=max_cols,
    )

    # Stage4: dispatch Decompile_xxx
    mode_all = ref.mode_all

    # 非 dialted 模式仍使用 record 自带 depth（保持你原来的行为）
    record_depth = ref.depth
    # dialted 模式使用新增输入参数 depth
    dialted_depth = depth

    if mode_all == 0:
        out_raw = Decompile_3_3_2s(data_in, record_depth) if stride == 2 else Decompile_3_3(data_in, record_depth)
        which = "Decompile_3_3_2s" if stride == 2 else "Decompile_3_3"
    elif mode_all == 1:
        out_raw = Decompile_1_3(data_in, record_depth)
        which = "Decompile_1_3"
    elif mode_all == 2:
        out_raw = Decompile_3_1(data_in, record_depth)
        which = "Decompile_3_1"
    elif mode_all == 3:
        out_raw = Decompile_dialted_1_2(data_in, dialted_depth)
        which = "Decompile_dialted_1_2"
    elif mode_all == 4:
        out_raw = Decompile_dialted_2_1(data_in, dialted_depth)
        which = "Decompile_dialted_2_1"
    elif mode_all == 5:
        out_raw = Decompile_gemm(data_in, a1_height)
        which = "Decompile_gemm"
    elif mode_all == 6:
        data_int = [[int(x, 2) for x in row] for row in data_in]
        out_raw = Decompile_6_6(data_int, record_depth)
        which = "Decompile_6_6"
    else:
        raise ValueError(f"mode_all 必须在 0..6，当前={mode_all}")

    print_matrix(out_raw, f"Stage4: {which} output (before depadding cut)", max_rows=max_rows, max_cols=max_cols)

    # Stage5: depadding 截断
    keep_rows = int(depadding_num)
    if isinstance(out_raw, tuple) and len(out_raw) == 2:
        C, out = out_raw
        out_cut = out[:keep_rows]
        out_after_cut: Any = (C, out_cut)
    else:
        out_after_cut = out_raw[:keep_rows]

    print_matrix(
        out_after_cut,
        "Stage5: Output after depadding cut (before optional dilated padding remove)",
        max_rows=max_rows,
        max_cols=max_cols,
    )

    def _remove_rightmost_col(mat: List[List[Any]]) -> List[List[Any]]:
        if not mat:
            return mat
        # 如果某些行为空，直接保持（或你也可以选择报错）
        return [row[:-1] if len(row) > 0 else row for row in mat]

    # ✅ Stage5.5：当 mode_all=3 或 4 且 padding_one_dilated=1 时，删除最右边一列
    out_after_dilated_remove: Any = out_after_cut
    if padding_one_dilated == 1 and mode_all in (3, 4):
        if isinstance(out_after_cut, tuple) and len(out_after_cut) == 2:
            C, mat = out_after_cut
            out_after_dilated_remove = (C, _remove_rightmost_col(mat))
        else:
            out_after_dilated_remove = _remove_rightmost_col(out_after_cut)

    print_matrix(
        out_after_dilated_remove,
        "Stage5.5: After optional dilated rightmost-col remove (before transpose)",
        max_rows=max_rows,
        max_cols=max_cols,
    )

    # Stage6: mode=1 / mode=4 最终输出前整体转置（保持你原来的判断条件）
    if mode_all in (1, 4):
        if isinstance(out_after_dilated_remove, tuple):
            C, mat = out_after_dilated_remove
            out_final = (C, [list(col) for col in zip(*mat)]) if mat else (C, [])
        else:
            mat = out_after_dilated_remove
            out_final = [list(col) for col in zip(*mat)] if mat else []

        print_matrix(out_final, "Stage6: Final output after transpose (mode=1 or 4)", max_rows=max_rows, max_cols=max_cols)
    else:
        out_final = out_after_dilated_remove
        print_matrix(out_final, "Stage6: Final output (no transpose needed)", max_rows=max_rows, max_cols=max_cols)

    return out_final


# ==============================================================================
# 你要的“按 core_num 把 group_id=0..max 的 data 打印出来” + 整组处理
# ==============================================================================

def print_core_group_data(core_num: int, dict_records: List[Dict[str, Any]], max_lines_per_gid: Optional[int] = None) -> None:
    print("\n" + "=" * 120)
    print(f"[INPUT BY CORE] core_num={core_num}")
    print(f"total_records_for_this_core = {len(dict_records)}")
    if not dict_records:
        return

    grouped: Dict[int, List[str]] = {}
    max_gid = -1
    for d in dict_records:
        gid = int(d.get("group_id"))
        max_gid = max(max_gid, gid)
        grouped.setdefault(gid, []).extend(list(d.get("data", [])))

    for gid in range(max_gid + 1):
        data = grouped.get(gid, [])
        print(f"\ncore_num={core_num}, group_id={gid}, rows={len(data)}")
        show = data if max_lines_per_gid is None else data[:max_lines_per_gid]
        for r, line in enumerate(show):
            print(f"  data[{r:04d}]: {line}")
        if max_lines_per_gid is not None and len(data) > max_lines_per_gid:
            print(f"  ... (+{len(data)-max_lines_per_gid} more rows)")


def Decompile_global_queues_debug(
    global_queues: List[List[Dict[str, Any]]],
    stride_list: List[int],
    depadding_num_list: List[int],
    depth_list: List[int],
    padding_one_dilated_list: List[int],
    *,
    max_rows: Optional[int] = 20,
    max_cols: Optional[int] = 16,
    max_lines_per_gid: Optional[int] = 6,
    a1_height:int,
) -> List[Any]:
    if len(global_queues) != len(stride_list) or len(global_queues) != len(depadding_num_list):
        raise ValueError("global_queues/stride_list/depadding_num_list 长度必须一致")

    outs: List[Any] = []
    for core_num, dict_records in enumerate(global_queues):
        # 先打印：core_num 下 gid=0..max 的 data
        print_core_group_data(core_num, dict_records, max_lines_per_gid=max_lines_per_gid)

        # 再整组送入 Decompile_all_debug（内部打印每个子函数中间结果）
        out_final = Decompile_all_debug(
            dict_records,
            stride=stride_list[core_num],
            depadding_num=depadding_num_list[core_num],
            depth=depth_list[core_num],
            padding_one_dilated=padding_one_dilated_list[core_num],
            core_num=core_num,
            max_rows=max_rows,
            max_cols=max_cols,
            a1_height = a1_height,
        )
        outs.append(out_final)

    return outs



#=====================================================================================================

def gen_data_transfer_config(
        core_num,
        is_act,
        row_write_depth,
        sram_write_choose
):
    """
    生成数据传输配置
    """
    opcode = '01'
    

    if core_num > 30 or core_num < 0:
        raise ValueError('core_num must be 0~7')
    core_num = format(core_num, '03b')

    if is_act > 1 or is_act < 0:
        raise ValueError('is_act must be 0~1')
    is_act = format(is_act, '01b')

    if row_write_depth > 512 or row_write_depth <= 0:
        raise ValueError('row_write_depth must be 1~512')
    row_write_depth = format(row_write_depth, '09b')

    if sram_write_choose > 7 or sram_write_choose < 0:
        raise ValueError('sram_write_choose must be 0~7')
    sram_write_choose = format(sram_write_choose, '03b')


    inst = opcode +'00' + core_num + is_act + row_write_depth + sram_write_choose + '00000000' + '0000'

    return inst

def gen_decision_config(
        cfg_ena,
        threshold_cfg,
        cfg_addr,
        t2ddl,
        workload,
        resource,
        k,
        threshold,
        in_weight_0,
        in_weight_1,
        in_weight_2,
        in_weight_3,
        in_step
):
    """
    生成调度配置
    """
    opcode = '11'
    
    if cfg_ena > 1 or cfg_ena < 0:
        raise ValueError('cfg_ena must be 0~1')
    cfg_ena = format(cfg_ena, '01b')

    if threshold_cfg > 1 or threshold_cfg < 0:
        raise ValueError('threshold_cfg must be 0~1')
    threshold_cfg = format(threshold_cfg, '01b')

    if cfg_addr > 7 or cfg_addr < 0:
        raise ValueError('cfg_addr must be 0~7')
    cfg_addr = format(cfg_addr, '03b')

    if t2ddl > 255 or t2ddl < 0:
        raise ValueError('t2ddl must be 0~255')
    t2ddl = format(t2ddl, '08b')

    if workload > 255 or workload < 0:
        raise ValueError('workload must be 0~255')
    workload = format(workload, '08b')

    if resource > 255 or resource < 0:
        raise ValueError('resource must be 0~255')
    resource = format(resource, '08b')

    if k > 255 or k < 0:
        raise ValueError('k must be 0~255')
    k = format(k, '08b')

    if threshold > 255 or threshold < 0:
        raise ValueError('threshold must be 0~255')
    threshold = format(threshold, '08b')

    if in_weight_0 > 255 or in_weight_0 < 0:
        raise ValueError('in_weight_0 must be 0~255')
    in_weight_0 = format(in_weight_0, '08b')

    if in_weight_1 > 255 or in_weight_1 < 0:
        raise ValueError('in_weight_1 must be 0~255')
    in_weight_1 = format(in_weight_1, '08b')

    if in_weight_2 > 255 or in_weight_2 < 0:
        raise ValueError('in_weight_2 must be 0~255')
    in_weight_2 = format(in_weight_2, '08b')

    if in_weight_3 > 255 or in_weight_3 < 0:
        raise ValueError('in_weight_3 must be 0~255')
    in_weight_3 = format(in_weight_3, '08b')

    if in_step > 255 or in_step < 0:
        raise ValueError('in_step must be 0~255')
    in_step = format(in_step, '08b')


    inst0 = opcode + cfg_ena + threshold_cfg + cfg_addr + t2ddl + workload + '0000000' + '00'
    inst1 = opcode + resource + k + threshold + '0000' + '01'
    inst1 = opcode + in_weight_0 + in_weight_1 + in_weight_2 + '0000' + '10'
    inst2 = opcode + in_weight_3 + in_step + '00000000' + '0000' + '11'

    return inst0,inst1,inst2

def gen_calculate_inst(
        core_num,
        mul_enable,#0:snn,1:cnn
        gemm_mode,#0:snn/ann 1:gemm
        kernel_size,#0:3x3 1:1*3
        step_mode,#0:4step 1:8step 2:12step
        row_read_depth
):
    """
    生成计算指令
    """
    opcode = "10"
    is_calculate = "1"

    if (core_num > 7 or core_num < 0) or core_num == 15:
        raise ValueError('core_num must be 0~7 or 15')
    core_num = format(core_num, '04b')
    
    if mul_enable == "snn":
        mul_enable = "0"
    elif mul_enable == "cnn":
        mul_enable = "1"
    else:
        raise ValueError('mul_enable must be "snn or cnn"')

    if gemm_mode == "conv":
        gemm_mode = "0"
    elif gemm_mode == "matrixmul":
        gemm_mode = "1"
    else:
        raise ValueError('gemm_mode must be "conv or matrixmul"')

    if kernel_size == "3*3":
        kernel_size = "0"
    elif kernel_size == "3*1":
        kernel_size = "1"
    elif kernel_size == "1*3":
        kernel_size = "1"
    elif kernel_size == "6*6":
        kernel_size = "0"
    else:
        raise ValueError('kernel_size must be "3*3 or 3*1 or 1*3 or 6*6"')
    
    if step_mode == "4":
        step_mode = "00"
    elif step_mode == "8":
        step_mode = "10"
    elif step_mode == "12":
        step_mode = "11"
    else:
        raise ValueError('step_mode must be "4 or 8 or 12"')

    if row_read_depth > 512 or row_read_depth <= 0:
        raise ValueError('row_read_depth must be 1~512')
    row_read_depth = format(row_read_depth, '09b')

    inst = opcode + is_calculate + '0' + core_num + mul_enable + gemm_mode + kernel_size + step_mode + row_read_depth + '00000000' + '00'

    return inst


def gen_data_back_inst(
    core_num,
    res_row_read_depth,
    sram_read_choose
):
    """
    生成数据回传指令
    """
    opcode = '10'
    is_calculate = '0'
    
    if core_num > 7 or core_num < 0:
        raise ValueError('core_num must be 0~7')
    core_num = format(core_num, '03b')
    
    if res_row_read_depth > 512 or res_row_read_depth <= 0:
        raise ValueError('res_row_read_depth must be 1~512')
    res_row_read_depth = format(res_row_read_depth, '09b')

    if sram_read_choose > 7 or sram_read_choose < 0:
        raise ValueError('sram_read_choose must be 0~7')
    sram_read_choose = format(sram_read_choose, '03b')
    
    inst = opcode + is_calculate + '00' + core_num + res_row_read_depth + sram_read_choose + '00000000' + '0000'

    return inst


def gen_inst_transfer_start_inst():
    inst = "10000000000000000000000000000000"
    return inst


def gen_inst_transfer_end_inst():
    inst = "10000000000000000000000000000001"
    return inst

def run_vcs_simulation(command_file="command.txt", output_file="output.txt"):
    """
    运行VCS仿真
    """
    print("开始运行VCS仿真...")
    
    # 检查命令文件是否存在
    if not os.path.exists(command_file):
        print(f"错误: 命令文件 {command_file} 不存在")
        return False
    
    try:
        # 第一步: 编译VCS
        # print("编译VCS...")
        # subprocess.run(["make", "vcs"], capture_output=True, text=True, check=True)
        # print("VCS编译完成")
        
        # 第二步: 运行仿真，传递command.txt作为参数
        print("运行仿真...")
        subprocess.run(["make", "software"], capture_output=True,text=True, check=True)
        print("仿真完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
        print(f"标准错误输出: {e.stderr}")
        return False
    except FileNotFoundError:
        print("错误: make命令未找到，请确保已安装make工具")
        return False

def get_serial_port_list():
    """
    寻找可用串口
    """
    port_list = []

    port_list_temp = list(serial.tools.list_ports.comports())
    if len(port_list_temp) == 0:
        print("[Warning] 无可用串口！")
    else:
        print("[Successful] 存在可用的串口，如下：")
        for my_port in port_list_temp:
            print(my_port)
            port_list.append(str(my_port))

    return port_list

def open_serial_port(portx, bps, timeout, stopbits, bytesize, parity):
    """
    打开串口
    """
    successful = False

    if parity == 'Even':
        my_parity = 'E'
    elif parity == 'Odd':
        my_parity = 'O'
    elif parity == 'Mark':
        my_parity = 'M'
    elif parity == 'Space':
        my_parity = 'S'
    else:  # default: 'None'
        my_parity = 'N'

    try:
        # 打开串口，并得到串口对象
        ser = serial.Serial(portx, bps, timeout=timeout, stopbits=stopbits, bytesize=bytesize, parity=my_parity)

        # 判断是否成功打开
        if (ser.is_open):
            successful = True
            th = threading.Thread(target=read_from_serial_port, args=(ser,))  # 创建一个子线程去等待读数据
            th.start()
    except Exception as e:
        print("open_serial_port error!", e)

    return ser, successful

def read_from_serial_port(ser):
    global data_list
    # 循环接收数据（此为死循环，可用线程实现）
    while True:
        if ser.in_waiting:
            data = b2a_hex(ser.read(ser.in_waiting)).decode('utf-8')  # 16进制的字符串，例如：'4141'，'FF'
            data_list.append(data)

def close_serial_port(ser):
    '''关闭串口'''
    global GLOBAL_NOTEND
    GLOBAL_NOTEND = False
    ser.close()
    print('串口已关闭！')

def swap_binary_parts(binary_str):
    """
    交换32位二进制字符串的指定部分。
    :param binary_str: 32位的二进制字符串
    :return: 修改后的32位二进制字符串
    """
    if len(binary_str) != 32:
        raise ValueError("Input string must be exactly 32 bits long.")
    # 将32位二进制字符串分为四个8位的部分
    part1 = binary_str[0:8]
    part2 = binary_str[8:16]
    part3 = binary_str[16:24]
    part4 = binary_str[24:32]
    # 交换位置：part1 <-> part4, part2 <-> part3
    swapped_str = part4 + part3 + part2 + part1
    
    return swapped_str

def swap_binary_parts_64(binary_str):
    """
    交换64位二进制字符串的指定部分。
    :param binary_str: 32位的二进制字符串
    :return: 修改后的32位二进制字符串
    """
    if len(binary_str) != 64:
        raise ValueError("Input string must be exactly 64 bits long.")
    # 将32位二进制字符串分为四个8位的部分
    part1 = binary_str[0:8]
    part2 = binary_str[8:16]
    part3 = binary_str[16:24]
    part4 = binary_str[24:32]
    part5 = binary_str[32:40]
    part6 = binary_str[40:48]
    part7 = binary_str[48:56]
    part8 = binary_str[56:64]
    swapped_str = part8 + part7 + part6 + part5 + part4 + part3 + part2 + part1
    
    return swapped_str

def binary_to_hex(binary_str, bits_per_hex_char=4):
    """
    将给定的二进制字符串转换为十六进制字符串，保证一定的宽度。
    
    :param binary_str: 二进制字符串
    :param bits_per_hex_char: 每个十六进制字符对应的位数，默认为4
    :return: 固定宽度的十六进制字符串
    """
    if not all(c in '01' for c in binary_str):
        raise ValueError("Input string must contain only '0' or '1'.")
    # 计算所需最小宽度（十六进制字符数）
    width = (len(binary_str) + bits_per_hex_char - 1) // bits_per_hex_char
    # 转换成整数
    int_value = int(binary_str, 2)
    # 使用format函数，指定最小宽度并用0填充
    hex_str = format(int_value, f'0{width}X')
    
    return hex_str

def write_to_serial_port(ser, text):
    """
    #向硬件写入
    """
    byte_num_sent = ser.write(a2b_hex(text))
    # print("[Successful] 向串口写入了" + str(byte_num_sent) + '个字节。')
    return byte_num_sent

def is_data_line(line):
    """
    检查行是否为数据行
    """
    stripped_line = line.strip()
    return all(c in '01 \n' for c in stripped_line) and stripped_line  # 包含换行符\n

def process_received_data(hex_data, filename):
    """
    处理接收到的16进制数据，转换为二进制并保存到文件
    
    Args:
        hex_data: 16进制字符串
        filename: 输出文件名
    """
    # 检查数据长度是否是16的倍数（每16个16进制字符=64bit）
    if len(hex_data) % 16 != 0:
        print(f"警告: 数据长度 {len(hex_data)} 不是16的倍数")
        print(hex_data)
        return
    
    # 分割为64bit块
    num_chunks = len(hex_data) // 16
    
    # 写入文件
    with open(filename, "w") as f:
        for i in range(num_chunks):
            # 提取64bit的16进制字符串
            hex_chunk = hex_data[i*16:(i+1)*16]
            
            # 将16进制转换为整数
            int_value = int(hex_chunk, 16)
            
            # 将整数转换为64bit二进制字符串（补零到64位）
            binary_str = bin(int_value)[2:].zfill(64)
            binary_str = swap_binary_parts_64(binary_str)


            
            # 写入文件（每行一个64bit二进制数）
            f.write(f"{binary_str}\n")
    
    print(f"已将 {num_chunks} 个64bit二进制数写入 {filename}")
     

def run_for_uart(ser,command_file="command.txt",output_file="output.txt"):
    """
    进行与硬件的交互
    """
    print("开始传输给硬件")
    global data_list
    data_list = []
    if not os.path.exists(command_file):
        print(f"错误: 命令与数据文件 {command_file} 不存在")
        return False

    try:
        #发送数据进入硬件    
        with open(command_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if is_data_line(line)]  
        for line in lines: 
            line = line.strip('\n')
            line = swap_binary_parts(line) #交换数据
            hex_line = binary_to_hex(line)
            write_to_serial_port(ser, hex_line)   #发送给下位机
        
        start_time = time.time()
        print("开始接收数据...")
        while True:
            if ser.in_waiting:
                start_time = time.time()  # 重置计时器
            elif time.time() - start_time > 3:
                break
        
        print(data_list)    
        print(ser.in_waiting)
        combined = "".join(data_list)  
        process_received_data(combined, output_file) 

    except KeyboardInterrupt:
        close_serial_port(ser)
        print("\nTransfer Error")

# def gen_once_calculate(num,depth,mode,kernel_size,step_mode):
#     '''  
#     num 使用到的核数 
#     write_depth 传输数据的长度 
#     mode 计算模式
#     kernel_size 卷积核大小
#     step_mode snn步长
#     '''                
#     print("===== 权重数据传输配置 =====")
#     for core_num in range(num):
#         inst = gen_data_transfer_config(core_num=core_num, is_act=0, row_write_depth=2)
#         print(inst)
#     print("===== 激活数据传输配置 =====")
#     for core_num in range(num):
#         inst = gen_data_transfer_config(core_num=core_num, is_act=1, row_write_depth=depth)
#         print(inst)
    
#     print("===== 指令传输 =====")
#     inst = gen_inst_transfer_start_inst()
#     print(inst)
#     for core_num in range(num):
#         if mode == "snn": 
#             inst = gen_calculate_inst(core_num=core_num,mul_enable = "snn",gemm_mode="conv",kernel_size = kernel_size,step_mode = step_mode,row_read_depth =depth)
#             print(inst)
#             inst = gen_data_back_inst(core_num=core_num, res_row_read_depth=(depth-2))
#             print(inst)
#         elif mode == "cnn": 
#             inst = gen_calculate_inst(core_num=core_num,mul_enable = "cnn",gemm_mode="conv",kernel_size = kernel_size,step_mode = step_mode,row_read_depth =depth)
#             print(inst)
#             inst = gen_data_back_inst(core_num=core_num, res_row_read_depth=(depth-2))
#             print(inst)
#         elif mode == "gemm": 
#             inst = gen_calculate_inst(core_num=core_num,mul_enable = "snn",gemm_mode="matrixmul",kernel_size = kernel_size,step_mode = step_mode,row_read_depth =depth)
#             print(inst)
#             inst = gen_data_back_inst(core_num=core_num, res_row_read_depth=1)
#             print(inst)
#         else : 
#             print("mode must be snn or cnn or gemm")
        
#     inst = gen_inst_transfer_end_inst()
#     print(inst)




def gen_once_calculate(num, group_max, group_list, depth_list, mode_list, kernel_size_list, step_mode_list, mode_all_list, inst_file="instructions.txt", command_file="command.txt",ser="COM3",using_mode="uart"):
    '''  
    num: 使用到的核数 
    groups: 组数
    depth_list: 每个核的传输数据长度列表 [depth_0, depth_1, ..., depth_{num-1}]
    mode_list: 每个核的计算模式列表 [mode_0, mode_1, ..., mode_{num-1}]
    kernel_size_list: 每个核的卷积核大小列表 [kernel_size_0, kernel_size_1, ...]
    step_mode_list: 每个核的snn步长列表 [step_mode_0, step_mode_1, ...]
    inst_file: 指令输出文件名
    command_file: 指令+数据混合输出文件名
    '''   
    # 参数验证
    if len(depth_list) != num:
        raise ValueError(f"depth_list长度应为{num}, 但得到{len(depth_list)}")
    if len(mode_list) != num:
        raise ValueError(f"mode_list长度应为{num}, 但得到{len(mode_list)}")
    if len(kernel_size_list) != num:
        raise ValueError(f"kernel_size_list长度应为{num}, 但得到{len(kernel_size_list)}")
    if len(step_mode_list) != num:
        raise ValueError(f"step_mode_list长度应为{num}, 但得到{len(step_mode_list)}")
    
    # 全局队列，存储num个核的结果队列
    global_queues: List[deque] = [deque() for _ in range(num)]
    # global_queues=[]
    # global_queues.append([])
    # 清空全局队列
    def clear_global_queues():
        """
        清空全局队列
        """
        for queue in global_queues:
            queue.clear()
        print("全局队列已清空")
    clear_global_queues()
    for group in range(group_max):
        with open(inst_file, 'w', encoding='utf-8') as f_inst,open(command_file, 'w', encoding='utf-8') as f_cmd:
            for core_num in range(num):
                if(group < group_list[core_num]):
                    quotient, remainder = divmod(core_num, 8)
                    # 读取权重数据
                    weight_file = f"w_{core_num}.txt"
                    with open(weight_file, 'r', encoding='utf-8') as data_f:
                        print("===== 权重数据传输配置 =====")
                        f_inst.write("===== 权重数据传输配置 =====\n")
                        f_cmd.write("===== 权重数据传输配置 =====\n")
                        inst = gen_data_transfer_config(core_num=quotient, is_act=0, row_write_depth=2,sram_write_choose = remainder)
                        print(inst)
                        f_inst.write(inst + '\n')
                        f_cmd.write(inst + '\n')

                        weight_data = data_f.readline()
                        f_cmd.write(f"===== 权重数据 (core {core_num}) =====\n")
                        f_cmd.write(weight_data)
                        print("===== 权重数据传输配置 =====")
                        f_inst.write("===== 权重数据传输配置 =====\n")
                        f_cmd.write("===== 权重数据传输配置 =====\n")
                        inst = gen_data_transfer_config(core_num=quotient, is_act=0, row_write_depth=2,sram_write_choose = remainder)
                        print(inst)
                        f_inst.write(inst + '\n')
                        f_cmd.write(inst + '\n')

                        weight_data = data_f.readline()
                        f_cmd.write(f"===== 权重数据 (core {core_num}) =====\n")
                        f_cmd.write(weight_data)

                
                if(group < group_list[core_num]):
                    print("===== 激活数据传输配置 =====")
                    f_inst.write("===== 激活数据传输配置 =====\n")
                    f_cmd.write("===== 激活数据传输配置 =====\n")
                    quotient, remainder = divmod(core_num, 8)
                    inst = gen_data_transfer_config(core_num=quotient, is_act=1, row_write_depth=depth_list[core_num],sram_write_choose = remainder)
                    print(inst)
                    f_inst.write(inst + '\n')
                    f_cmd.write(inst + '\n')
                    
                    # 读取激活数据
                    act_file = f"a_{core_num}_{(group+1)}.txt"

                    with open(act_file, 'r', encoding='utf-8') as data_f:
                        act_data = data_f.read()
                        f_cmd.write(f"===== 激活数据 (core {core_num}) =====\n")
                        f_cmd.write(act_data)
                        if not act_data.endswith('\n'):
                            f_cmd.write('\n')
            
            
                

            
            print("===== 指令传输 =====")
            f_inst.write("===== 指令传输 =====\n")
            f_cmd.write("===== 指令传输 =====\n")
            
            inst = gen_inst_transfer_start_inst()
            print(inst)
            f_inst.write(inst + '\n')
            f_cmd.write(inst + '\n')
            
            for core_num in range(num):
                if(group < group_list[core_num]):
                    quotient, remainder = divmod(core_num, 8)
                    if remainder == 0:
                        if mode_list[core_num] == "snn": 
                            inst = gen_calculate_inst(core_num=quotient, mul_enable="snn", gemm_mode="conv", 
                                                    kernel_size=kernel_size_list[core_num], step_mode=step_mode_list[core_num], row_read_depth=depth_list[core_num] + 2 )
                            print(inst)
                            f_inst.write(inst + '\n')
                            f_cmd.write(inst + '\n')
                            
                        elif mode_list[core_num] == "cnn": 
                            inst = gen_calculate_inst(core_num=quotient, mul_enable="cnn", gemm_mode="conv", 
                                                    kernel_size=kernel_size_list[core_num], step_mode="4", row_read_depth=depth_list[core_num] + 2) 
                            print(inst)
                            f_inst.write(inst + '\n')
                            f_cmd.write(inst + '\n')
                            
                        elif mode_list[core_num] == "gemm": 
                            inst = gen_calculate_inst(core_num=quotient, mul_enable="cnn", gemm_mode="matrixmul", 
                                                    kernel_size=kernel_size_list[core_num], step_mode="4", row_read_depth=depth_list[core_num] + 2)
                            print(inst)
                            f_inst.write(inst + '\n')
                            f_cmd.write(inst + '\n')
                            
                        else: 
                            error_msg = "mode must be snn or cnn or gemm"
                            print(error_msg)
                            f_inst.write(error_msg + '\n')
                            f_cmd.write(error_msg + '\n')
            
            for core_num in range(num):
                if(group < group_list[core_num]):
                    quotient, remainder = divmod(core_num, 8)
                    if mode_list[core_num] == "snn": 
                        inst = gen_data_back_inst(core_num=quotient, res_row_read_depth=(depth_list[core_num] - 2),sram_read_choose = remainder)
                        print(inst)
                        f_inst.write(inst + '\n')
                        f_cmd.write(inst + '\n')
                        
                    elif mode_list[core_num] == "cnn": 
                        inst = gen_data_back_inst(core_num=quotient, res_row_read_depth=(depth_list[core_num] - 2),sram_read_choose = remainder)
                        print(inst)
                        f_inst.write(inst + '\n')
                        f_cmd.write(inst + '\n')
                        
                    elif mode_list[core_num] == "gemm": 
                        inst = gen_data_back_inst(core_num=quotient, res_row_read_depth = 1,sram_read_choose = remainder)
                        print(inst)
                        f_inst.write(inst + '\n')
                        f_cmd.write(inst + '\n')
                    else: 
                        error_msg = "mode must be snn or cnn or gemm"
                        print(error_msg)
                        f_inst.write(error_msg + '\n')
                        f_cmd.write(error_msg + '\n')

            inst = gen_inst_transfer_end_inst()
            print(inst)
            f_inst.write(inst + '\n')
            f_cmd.write(inst + '\n')
        
        print(f"指令文件已生成: {inst_file}")
        print(f"混合文件已生成: {command_file}")

        if(using_mode == "uart"):
            #通过uart与硬件交互 
            run_for_uart(ser=ser,command_file="command.txt",output_file="output.txt")
        elif(using_mode == "vcs"):
            # 运行VCS
            success = run_vcs_simulation("command.txt", "output.txt")
            if success:
                print("仿真成功完成")
            else:
                print("仿真失败")

        print("group_list",group_list)
        print("depth_list",depth_list)
        print("mode_list",mode_list)
        print("kernel_size_list",kernel_size_list)
        print("step_mode_list",step_mode_list)
        #保存每组的结果到全局队列中
        output_file = f"output.txt"
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        print(f"读取文件 {output_file}: 共 {len(lines)} 行64bit结果")
        current_line = 0

        for core_num in range(num):
            if(group < group_list[core_num]):
                if mode_list[core_num] == "gemm":
                    lines_to_take = 1
                elif mode_list[core_num] == "cnn":
                    lines_to_take = depth_list[core_num] - 2
                elif mode_list[core_num] == "snn":
                    lines_to_take = depth_list[core_num] - 2
                else:
                    print(f"mode must be snn or cnn or gemm")
                
                # 检查是否有足够的行
                if current_line + lines_to_take > len(lines):
                    print(f"警告: 核{core_num}需要{lines_to_take}行，但只剩{len(lines)-current_line}行")
                    lines_to_take = len(lines) - current_line
                    if lines_to_take <= 0:
                        break
                
                # 提取该核的结果
                core_data = lines[current_line:current_line + lines_to_take]
                print("group",group)
                print("core_data",core_data)
                print("current_line",current_line)
                print("lines_to_take",lines_to_take)
                global_queues[core_num].append({
                    'group_id': group,
                    'mode': mode_list[core_num],
                    'depth': depth_list[core_num],
                    'step_mode':step_mode_list[core_num],
                    'mode_all':mode_all_list[core_num],
                    'data': core_data,
                    'core_num': core_num,
                })
                current_line += lines_to_take
    print("global_queues",global_queues)
    return global_queues


#===================================================================================================

def run_decompile_all_for_all_cores(global_queues, stride: int, depadding_num: int, depth: int, padding_one_dilated: int,a1_height: int):
    outs = []

    for core_num in range(len(global_queues)):
        records = global_queues[core_num]  # ✅ 该 core 的所有 group 记录（List[dict]）

        if not records:
            print(f"[SKIP] core_num={core_num} has no records.")
            outs.append(None)
            continue

        out = Decompile_all_debug(records, stride=stride, depadding_num=depadding_num, depth=depth,padding_one_dilated=padding_one_dilated,a1_height=a1_height)
        outs.append(out)

        print(f"[DONE] core_num={core_num}, num_records={len(records)}")

    return outs

#=======================================主函数================================================


def calculate_using_hardware(
    #目前测试：所有core对相同的特征图进行计算
    A_list, # 激活矩阵
    A2_list, # gemm的第二个矩阵
    W_list, # 权重矩阵，与A2_list为相同输入即可
    kernel_size_in_list, # 3*3, 1*3, 3*1, 6*6(先不考虑6*6)
    dialation_in_list, # 1*2, 2*1, 1*1
    step_mode_in_list, # 4, 8, 12
    scale_list, # 28bit 缩放因子
    pe_mode_in_list, #cnn, snn, gemm
    kernel_num, # 使用核数(目前核数最大为30个左右)
    a1_height, # 激活矩阵(A_list)的行数
    stride: int, # 步长(int值，非列表形式)
    depadding_num: int, # 理论上,最终输出的特征图的行数(不包括软件做的反padding,即传给软件的尺寸,int值,非列表形式)
    depth: int, # 新加的depth，与奇偶数有关
    padding_one_dilated: int,
    inst_file: str = "instructions.txt",
    command_file: str = "command.txt",
    ser="COM3",
    using_mode: str = "uart" #选择uart或者vcs
):
    (
        *outs,
        kernel_num_out,
        mode_list,
        depth_list,
        pe_mode_list,
        step_mode_list,
        kernel_size_list,
        group_list,
        group_max
    ) = transform_all_1(
        A_list, A2_list, W_list,
        kernel_size_in_list,
        dialation_in_list,
        step_mode_in_list,
        scale_list,
        pe_mode_in_list,
        kernel_num
    )

    global_queues = gen_once_calculate(
        num=kernel_num_out,
        group_max=group_max,
        group_list=group_list,
        depth_list=depth_list,
        mode_list=pe_mode_list,
        kernel_size_list=kernel_size_list,
        step_mode_list=step_mode_list,
        mode_all_list=mode_list,
        inst_file=inst_file,
        command_file=command_file,
        ser=ser,
        using_mode=using_mode
    )

    outs = run_decompile_all_for_all_cores(
        global_queues,
        stride=stride,
        depadding_num=depadding_num,
        depth=depth,
        padding_one_dilated=padding_one_dilated,
        a1_height= a1_height,
    )
    return outs

#################################################测试代码###############################################

import numpy as np

def _ensure_file_exists(path: str, content: str = "") -> None:
    # gen_once_calculate 如果要读文件，文件至少得存在
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def make_int4_matrix(h, w, seed=0):
    rng = np.random.default_rng(seed)
    # int4 常见范围
    return rng.integers(0, 15, size=(h, w), dtype=np.int32)

def test_calculate_using_hardware(ser,using_mode):
    np.random.seed(0)

    def rand_bin4_matrix(shape, max_val: int):
        x = np.random.randint(0, max_val + 1, size=shape)
        return np.vectorize(lambda v: format(int(v), "04b"))(x)

    _MAP_SNN_4  = {"0000":"0000","0001":"0001","0010":"0011","0011":"0111","0100":"1111"}
    _MAP_SNN_8  = {"0000":"0000","0001":"0001","0010":"0011","0011":"0010","0100":"0110","0101":"0111","0110":"0101","0111":"0100","1000":"1100"}
    _MAP_SNN_12 = {"0000":"0000","0001":"0001","0010":"0011","0011":"0010","0100":"0110","0101":"0111","0110":"0101","0111":"0100",
                   "1000":"1100","1001":"1101","1010":"1111","1011":"1110","1100":"1010"}

    def pre_transform_A_for_print(A, pe_mode_in, step_mode_in_str):
        pe = int(pe_mode_in)
        if pe in (0, 2):  # ann/gemm
            A_arr = np.array(A, dtype=str)
            return np.char.zfill(A_arr, 4)

        step = int(step_mode_in_str)
        table = _MAP_SNN_4 if step == 4 else (_MAP_SNN_8 if step == 8 else (_MAP_SNN_12 if step == 12 else None))
        if table is None:
            raise ValueError(f"snn 仅支持 step_mode_in='4'/'8'/'12'，但收到 {step_mode_in_str}")

        A_arr = np.char.zfill(np.array(A, dtype=str), 4)
        out = np.empty(A_arr.shape, dtype="<U4")
        it = np.nditer(A_arr, flags=["multi_index", "refs_ok"])
        for x in it:
            s = str(x.item()).strip()
            if s not in table:
                raise ValueError(f"预变换映射表中找不到输入值 '{s}' (snn, step={step})")
            out[it.multi_index] = table[s]
        return out

    def print_scale_28b(scale: int):
        print("scale(28b) =", format(int(scale), "028b"))
    # ====== 多路测试样例（保持原来风格）======
    Y_big, X_big = 8, 16
    Y_gemm, X_gemm = 3, 5
    Y2_gemm = 3
    Y_mode6 = 16
    X_mode6 = 7

    scales = [0x0000001, 0x0000001,0x0000001,0x0000001, 0x0000001,0x0000001,0x0000001, 0x0000001,0x0000001,0x0000001, 0x0000001,0x0000001,0x0000001, 0x0000001,0x0000001,0x0000001, 0x0000001,0x0000001,0x0000001, 0x0000001,0x0000001]

        # -------- 第0组：路0、路1、路2 -> 采用原路0模式 --------
    # 路0
    A0  = rand_bin4_matrix((Y_big, X_big), max_val=4)
    print("A0",A0)
    A20 = np.zeros((X_big, Y_big), dtype=int)
    pe_mode_in0, dialation_in0, kernel_size_in0 = 0, "1*1", "3*3"
    if kernel_size_in0 == "1*3" or kernel_size_in0 == "3*1":
        # 先生成一行 3 个随机 4bit 数
        row = np.random.randint(0, 16, size=(1, 3))
        # 复制成 3 行
        W0 = np.repeat(row, 3, axis=0)
    else:
        # 完全随机的 3x3
        W0 = np.random.randint(0, 16, size=(3, 3))
    step_mode_in0, scale0 = "4", scales[0]
    A0_pre = pre_transform_A_for_print(A0, pe_mode_in0, step_mode_in0)
    # 路1（使用原路0模式）
    A1  = rand_bin4_matrix((Y_big, X_big), max_val=4)
    A21 = np.zeros((X_big, Y_big), dtype=int)
    pe_mode_in1, dialation_in1, kernel_size_in1 = 0, "1*1", "3*3"
    if kernel_size_in1 == "1*3" or kernel_size_in1 == "3*1":
        # 先生成一行 3 个随机 4bit 数
        row = np.random.randint(0, 16, size=(1, 3))
        # 复制成 3 行
        W1 = np.repeat(row, 3, axis=0)
    else:
        # 完全随机的 3x3
        W1 = np.random.randint(0, 16, size=(3, 3))
    step_mode_in1, scale1 = "4", scales[1]
    A1_pre = pre_transform_A_for_print(A1, pe_mode_in1, step_mode_in1)

    # 路2（使用原路0模式）
    A2  = rand_bin4_matrix((Y_big, X_big), max_val=4)
    A22 = np.zeros((X_big, Y_big), dtype=int)
    pe_mode_in2, dialation_in2, kernel_size_in2 = 0, "1*1", "3*3"
    if kernel_size_in2 == "1*3" or kernel_size_in2 == "3*1":
        # 先生成一行 3 个随机 4bit 数
        row = np.random.randint(0, 16, size=(1, 3))
        # 复制成 3 行
        W2 = np.repeat(row, 3, axis=0)
    else:
        # 完全随机的 3x3
        W2 = np.random.randint(0, 16, size=(3, 3))
    step_mode_in2, scale2 = "4", scales[2]
    A2_pre = pre_transform_A_for_print(A2, pe_mode_in2, step_mode_in2)

    # -------- 第1组：路3、路4、路5 -> 采用原路1模式 --------
    # 路3（使用原路1模式）
    A3  = rand_bin4_matrix((Y_big, X_big), max_val=8)
    A23 = np.zeros((X_big, Y_big), dtype=int)
    pe_mode_in3, dialation_in3, kernel_size_in3 = 1, "1*1", "3*1"
    if kernel_size_in3 == "1*3" or kernel_size_in3 == "3*1":
        # 先生成一行 3 个随机 4bit 数
        row = np.random.randint(0, 16, size=(1, 3))
        # 复制成 3 行
        W3 = np.repeat(row, 3, axis=0)
    else:
        # 完全随机的 3x3
        W3 = np.random.randint(0, 16, size=(3, 3))
    step_mode_in3, scale3 = "8", scales[3]
    A3_pre = pre_transform_A_for_print(A3, pe_mode_in3, step_mode_in3)

    # 路4（使用原路1模式）
    A4  = rand_bin4_matrix((Y_big, X_big), max_val=8)
    A24 = np.zeros((X_big, Y_big), dtype=int)
    pe_mode_in4, dialation_in4, kernel_size_in4 = 1, "1*1", "3*1"
    if kernel_size_in4 == "1*3" or kernel_size_in4 == "3*1":
        # 先生成一行 3 个随机 4bit 数
        row = np.random.randint(0, 16, size=(1, 3))
        # 复制成 3 行
        W4 = np.repeat(row, 3, axis=0)
    else:
        # 完全随机的 3x3
        W4 = np.random.randint(0, 16, size=(3, 3))
    step_mode_in4, scale4 = "8", scales[4]
    A4_pre = pre_transform_A_for_print(A4, pe_mode_in4, step_mode_in4)

    # 路5（使用原路1模式）
    A5  = rand_bin4_matrix((Y_big, X_big), max_val=8)
    A25 = np.zeros((X_big, Y_big), dtype=int)
    W5 = np.repeat(np.random.randint(0, 16, size=(1, 3)), 3, axis=0)
    pe_mode_in5, dialation_in5, kernel_size_in5 = 1, "1*1", "3*1"
    step_mode_in5, scale5 = "8", scales[5]
    A5_pre = pre_transform_A_for_print(A5, pe_mode_in5, step_mode_in5)

    # -------- 第2组：路6、路7、路8 -> 采用原路2模式 --------
    # 路6（使用原路2模式）
    A6  = rand_bin4_matrix((Y_big, X_big), max_val=4)
    A26 = np.zeros((X_big, Y_big), dtype=int)
    W6 = np.repeat(np.random.randint(0, 16, size=(1, 3)), 3, axis=0)
    pe_mode_in6, dialation_in6, kernel_size_in6 = 0, "1*1", "1*3"
    step_mode_in6, scale6 = "12", scales[6]
    A6_pre = pre_transform_A_for_print(A6, pe_mode_in6, step_mode_in6)

    # 路7（使用原路2模式）
    A7  = rand_bin4_matrix((Y_big, X_big), max_val=4)
    A27 = np.zeros((X_big, Y_big), dtype=int)
    W7 = np.repeat(np.random.randint(0, 16, size=(1, 3)), 3, axis=0)
    pe_mode_in7, dialation_in7, kernel_size_in7 = 0, "1*1", "1*3"
    step_mode_in7, scale7 = "12", scales[7]
    A7_pre = pre_transform_A_for_print(A7, pe_mode_in7, step_mode_in7)

    # 路8（使用原路2模式）
    A8  = rand_bin4_matrix((Y_big, X_big), max_val=4)
    A28 = np.zeros((X_big, Y_big), dtype=int)
    W8 = np.repeat(np.random.randint(0, 16, size=(1, 3)), 3, axis=0)
    pe_mode_in8, dialation_in8, kernel_size_in8 = 0, "1*1", "1*3"
    step_mode_in8, scale8 = "12", scales[8]
    A8_pre = pre_transform_A_for_print(A8, pe_mode_in8, step_mode_in8)

    # -------- 第3组：路9、路10、路11 -> 采用原路3模式 --------
    # 路9（使用原路3模式）
    A9  = rand_bin4_matrix((Y_big, X_big), max_val=4)
    A29 = np.zeros((X_big, Y_big), dtype=int)
    W9 = np.repeat(np.random.randint(0, 16, size=(1, 3)), 3, axis=0)
    pe_mode_in9, dialation_in9, kernel_size_in9 = 0, "1*2", "1*3"
    step_mode_in9, scale9 = "4", scales[9]
    A9_pre = pre_transform_A_for_print(A9, pe_mode_in9, step_mode_in9)

    # 路10（使用原路3模式）
    A10  = rand_bin4_matrix((Y_big, X_big), max_val=4)
    A210 = np.zeros((X_big, Y_big), dtype=int)
    W10 = np.repeat(np.random.randint(0, 16, size=(1, 3)), 3, axis=0)
    pe_mode_in10, dialation_in10, kernel_size_in10 = 0, "1*2", "1*3"
    step_mode_in10, scale10 = "4", scales[10]
    A10_pre = pre_transform_A_for_print(A10, pe_mode_in10, step_mode_in10)

    # 路11（使用原路3模式）
    A11  = rand_bin4_matrix((Y_big, X_big), max_val=4)
    A211 = np.zeros((X_big, Y_big), dtype=int)
    W11 = np.repeat(np.random.randint(0, 16, size=(1, 3)), 3, axis=0)
    pe_mode_in11, dialation_in11, kernel_size_in11 = 0, "1*2", "1*3"
    step_mode_in11, scale11 = "4", scales[11]
    A11_pre = pre_transform_A_for_print(A11, pe_mode_in11, step_mode_in11)

    # -------- 第4组：路12、路13、路14 -> 采用原路4模式 --------
    # 路12（使用原路4模式）
    A12  = rand_bin4_matrix((Y_big, X_big), max_val=8)
    A212 = np.zeros((X_big, Y_big), dtype=int)
    W12 = np.repeat(np.random.randint(0, 16, size=(1, 3)), 3, axis=0)
    pe_mode_in12, dialation_in12, kernel_size_in12 = 1, "2*1", "3*1"
    step_mode_in12, scale12 = "8", scales[12]
    A12_pre = pre_transform_A_for_print(A12, pe_mode_in12, step_mode_in12)

    # 路13（使用原路4模式）
    A13  = rand_bin4_matrix((Y_big, X_big), max_val=8)
    A213 = np.zeros((X_big, Y_big), dtype=int)
    W13 = np.repeat(np.random.randint(0, 16, size=(1, 3)), 3, axis=0)
    pe_mode_in13, dialation_in13, kernel_size_in13 = 1, "2*1", "3*1"
    step_mode_in13, scale13 = "8", scales[13]
    A13_pre = pre_transform_A_for_print(A13, pe_mode_in13, step_mode_in13)

    # 路14（使用原路4模式）
    A14  = rand_bin4_matrix((Y_big, X_big), max_val=8)
    A214 = np.zeros((X_big, Y_big), dtype=int)
    W14 = np.repeat(np.random.randint(0, 16, size=(1, 3)), 3, axis=0)
    pe_mode_in14, dialation_in14, kernel_size_in14 = 1, "2*1", "3*1"
    step_mode_in14, scale14 = "8", scales[14]
    A14_pre = pre_transform_A_for_print(A14, pe_mode_in14, step_mode_in14)

    # -------- 第5组：路15、路16、路17 -> 采用原路5模式（GEMM）--------
    # 路15（使用原路5模式 - GEMM）
    A15  = rand_bin4_matrix((Y_gemm, X_gemm), max_val=15)
    A215 = np.random.randint(0, 16, size=(X_gemm, Y2_gemm))
    W15  = np.random.randint(0, 16, size=(3, 3))
    pe_mode_in15, dialation_in15, kernel_size_in15 = 2, "1*1", "3*3"
    step_mode_in15, scale15 = "12", scales[15]
    A15_pre = pre_transform_A_for_print(A15, pe_mode_in15, step_mode_in15)

    # 路16（使用原路5模式 - GEMM）
    A16  = rand_bin4_matrix((Y_gemm, X_gemm), max_val=15)
    A216 = np.random.randint(0, 16, size=(X_gemm, Y2_gemm))
    W16  = np.random.randint(0, 16, size=(3, 3))
    pe_mode_in16, dialation_in16, kernel_size_in16 = 2, "1*1", "3*3"
    step_mode_in16, scale16 = "12", scales[16]
    A16_pre = pre_transform_A_for_print(A16, pe_mode_in16, step_mode_in16)

    # 路17（使用原路5模式 - GEMM）
    A17  = rand_bin4_matrix((Y_gemm, X_gemm), max_val=15)
    A217 = np.random.randint(0, 16, size=(X_gemm, Y2_gemm))
    W17  = np.random.randint(0, 16, size=(3, 3))
    pe_mode_in17, dialation_in17, kernel_size_in17 = 2, "1*1", "3*3"
    step_mode_in17, scale17 = "12", scales[17]
    A17_pre = pre_transform_A_for_print(A17, pe_mode_in17, step_mode_in17)

    # # -------- 第6组：路18、路19、路20 -> 采用原路6模式（6*6）--------
    # # 路18（使用原路6模式 - 6*6）
    # A18  = rand_bin4_matrix((Y_mode6, X_mode6), max_val=4)
    # A218 = np.zeros((X_mode6, Y_mode6), dtype=int)
    # W18  = np.random.randint(0, 16, size=(6, 6))
    # pe_mode_in18, dialation_in18, kernel_size_in18 = 0, "1*1", "6*6"
    # step_mode_in18, scale18 = "4", scales[18]
    # A18_pre = pre_transform_A_for_print(A18, pe_mode_in18, step_mode_in18)

    # # 路19（使用原路6模式 - 6*6）
    # A19  = rand_bin4_matrix((Y_mode6, X_mode6), max_val=4)
    # A219 = np.zeros((X_mode6, Y_mode6), dtype=int)
    # W19  = np.random.randint(0, 16, size=(6, 6))
    # pe_mode_in19, dialation_in19, kernel_size_in19 = 0, "1*1", "6*6"
    # step_mode_in19, scale19 = "4", scales[19]
    # A19_pre = pre_transform_A_for_print(A19, pe_mode_in19, step_mode_in19)

    # # 路20（使用原路6模式 - 6*6）
    # A20  = rand_bin4_matrix((Y_mode6, X_mode6), max_val=4)
    # A220 = np.zeros((X_mode6, Y_mode6), dtype=int)
    # W20  = np.random.randint(0, 16, size=(6, 6))
    # pe_mode_in20, dialation_in20, kernel_size_in20 = 0, "1*1", "6*6"
    # step_mode_in20, scale20 = "4", scales[20]
    # A20_pre = pre_transform_A_for_print(A20, pe_mode_in20, step_mode_in20)

    A_list              = [A0,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15,A16,A17]
    A2_list             = [A20,A21,A22,A23,A24,A25,A26,A27,A28,A29,A210,A211,A212,A213,A214,A215,A216,A217]
    W_list              = [W0,W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,W12,W13,W14,W15,W16,W17]
    kernel_size_in_list = [kernel_size_in0,kernel_size_in1,kernel_size_in2,kernel_size_in3,kernel_size_in4,kernel_size_in5,kernel_size_in6,kernel_size_in7,kernel_size_in8,kernel_size_in9,kernel_size_in10,kernel_size_in11,kernel_size_in12,kernel_size_in13,kernel_size_in14,kernel_size_in15,kernel_size_in16,kernel_size_in17]
    dialation_in_list   = [dialation_in0,dialation_in1,dialation_in2,dialation_in3,dialation_in4,dialation_in5,dialation_in6,dialation_in7,dialation_in8,dialation_in9,dialation_in10,dialation_in11,dialation_in12,dialation_in13,dialation_in14,dialation_in15,dialation_in16,dialation_in17]
    step_mode_in_list   = [step_mode_in0,step_mode_in1,step_mode_in2,step_mode_in3,step_mode_in4,step_mode_in5,step_mode_in6,step_mode_in7,step_mode_in8,step_mode_in9,step_mode_in10,step_mode_in11,step_mode_in12,step_mode_in13,step_mode_in14,step_mode_in15,step_mode_in16,step_mode_in17]
    scale_list          = [scale0,scale1,scale2,scale3,scale4,scale5,scale6,scale7,scale8,scale9,scale10,scale11,scale12,scale13,scale14,scale15,scale16,scale17]
    pe_mode_in_list     = [pe_mode_in0,pe_mode_in1,pe_mode_in2,pe_mode_in3,pe_mode_in4,pe_mode_in5,pe_mode_in6,pe_mode_in7,pe_mode_in8,pe_mode_in9,pe_mode_in10,pe_mode_in11,pe_mode_in12,pe_mode_in13,pe_mode_in14,pe_mode_in15,pe_mode_in16,pe_mode_in17]

    kernel_num = 18
    stride = 1
    depadding_num = 20
    inst_file = "instructions.txt"
    command_file = "command.txt"
    outs = calculate_using_hardware(
        A_list=A_list,
        A2_list=A2_list,
        W_list=W_list,
        kernel_size_in_list=kernel_size_in_list,
        dialation_in_list=dialation_in_list,
        step_mode_in_list=step_mode_in_list,
        scale_list=scale_list,
        pe_mode_in_list=pe_mode_in_list,
        kernel_num=kernel_num,
        stride=stride,
        depadding_num=depadding_num,
        inst_file=inst_file,
        command_file=command_file,
        a1_height=Y_gemm,
        ser=ser,
        using_mode=using_mode
    )

    print("\n===== OUTPUT =====")
    # outs 结构取决于 run_decompile_all_for_all_cores 的返回
    # 这里先通用打印
    if isinstance(outs, (list, tuple)):
        print("outs type =", type(outs), "len =", len(outs))
        for idx, item in enumerate(outs):
            print(f"[{idx}] type={type(item)}")
            try:
                if hasattr(item, "shape"):
                    print("   shape:", item.shape)
                else:
                    # 避免太长：只显示前一点
                    s = str(item)
                    print("   value:", s[:300] + ("..." if len(s) > 300 else ""))
            except Exception as e:
                print("   (print failed)", e)
    else:
        print("outs type =", type(outs))
        print(str(outs)[:500])




if __name__ == "__main__":
    #主函数中加入uart的整体配置信息
    using_mode = "uart" #选择使用uart："uart"或者vcs:"vcs"或者vcs:"vcs_uart"(前两种由python产生指令，最后一种从uvm拿到指令和数据，通过uart测试,用于测试uvm中的测试样例在fpga上的正确性)
    if using_mode == "uart" :
        try:

            # 获取可用的串口的列表
            port_list = get_serial_port_list()

            # 打开列表中的某个串口，进行数据收发
            if port_list:
                # 选择串口
                while True:
                    portx_in_port_list = False
                    portx = 'COM3'  #input("请输入要打开的串口的名称(例如：COM5)：")
                    

                    for my_port in port_list:
                        if portx == my_port[:4]:
                            portx_in_port_list = True
                            break
                    if portx_in_port_list:
                        break
                # 设置其余参数
                
                bps = 115200
                timeout = 1
                stopbits = 1
                bytesize = 8
                parity = 'Odd'
                # 打开串口
                ser, successful = open_serial_port(portx, bps, timeout, stopbits, bytesize, parity)
                
                # 写配置和数据
                if successful == True:
                    test_calculate_using_hardware(ser=ser,using_mode=using_mode)

        except KeyboardInterrupt:
            close_serial_port(ser)
            print("\nApplication exit!")

    elif using_mode == "vcs_uart" :
        try:

            # 获取可用的串口的列表
            port_list = get_serial_port_list()

            # 打开列表中的某个串口，进行数据收发
            if port_list:
                # 选择串口
                while True:
                    portx_in_port_list = False
                    portx = 'COM3'  #input("请输入要打开的串口的名称(例如：COM5)：")
                    

                    for my_port in port_list:
                        if portx == my_port[:4]:
                            portx_in_port_list = True
                            break
                    if portx_in_port_list:
                        break
                # 设置其余参数
                
                bps = 115200
                timeout = 1
                stopbits = 1
                bytesize = 8
                parity = 'Odd'
                # 打开串口
                ser, successful = open_serial_port(portx, bps, timeout, stopbits, bytesize, parity)
                
                # 写配置和数据
                if successful == True:
                    run_for_uart(ser=ser,command_file="command.txt",output_file="output.txt")
                    #command.txt从uvm处拿到，output.txt存储结果，与uvm跑出来的期望结果进行对比

        except KeyboardInterrupt:
            close_serial_port(ser)
            print("\nApplication exit!")










