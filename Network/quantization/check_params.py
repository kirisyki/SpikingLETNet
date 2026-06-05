import torch

def value_distribution_perchannel(x_q: torch.Tensor, num_bins: int = 8):
    """
    统计量化张量每个通道的取值概率分布。

    Args:
        x_q: 量化后的张量，shape = [T, B, C, H, W]，值范围应为 0 ~ num_bins-1
        num_bins: 量化 bins 数量。0~7 对应 num_bins=8。

    Returns:
        prob: shape = [C, num_bins]
              prob[c, v] 表示第 c 个通道中取值 v 的概率
        count: shape = [C, num_bins]
               count[c, v] 表示第 c 个通道中取值 v 的数量
    """
    assert x_q.dim() == 5, f"Expected x_q shape [T, B, C, H, W], got {x_q.shape}"

    T, B, C, H, W = x_q.shape

    # 调整成 [C, T, B, H, W]，再展平为 [C, N]
    x_flat = x_q.permute(2, 0, 1, 3, 4).reshape(C, -1)

    # count[c, v] = 第 c 个通道中值为 v 的数量
    count = torch.stack([
        torch.bincount(x_flat[c].long(), minlength=num_bins)
        for c in range(C)
    ], dim=0)

    # 每个通道的总元素数
    total = x_flat.shape[1]

    prob = count.float() / total

    return prob, count