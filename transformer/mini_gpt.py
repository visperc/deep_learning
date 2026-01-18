import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    最小可用的多头因果自注意力 (Multi-Head Causal Self-Attention)
    输入:  x 形状 (B, T, C)   B=batch, T=序列长度, C=d_model
    输出:  y 形状 (B, T, C)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # 每个 head 的维度 d_k

        # 关键点：工程上通常用一个线性层一次性算出 Q,K,V，再切分
        # 这样更快（一次 GEMM），也更贴近主流实现
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        # 输出投影：把 concat 后的多头结果映射回 d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 注册一个“因果 mask”缓存（最大长度先不固定也行，这里给一个上限示例）
        # 你也可以在 forward 里动态生成；缓存的好处是避免每次构造。
        self.register_buffer("mask", None, persistent=False)

    def _get_causal_mask(self, T: int, device):
        """
        生成 (1, 1, T, T) 的上三角 mask（未来位置为 -inf）
        为什么是这个形状：为了能 broadcast 到 (B, n_heads, T, T)
        """
        if self.mask is None or self.mask.size(-1) < T:
            # torch.triu 生成上三角（含对角线以上），diagonal=1 表示严格未来
            m = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
            # 用 -inf 屏蔽未来注意力，softmax 后概率为 0
            self.mask = m
        return self.mask[:T, :T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # 拿到 batch/seq/hidden
        assert C == self.d_model

        # 1) 一次线性映射得到 qkv，形状 (B, T, 3C)
        # 为什么：Wq/Wk/Wv 本质是可学习线性投影；合并为一个矩阵乘更高效
        qkv = self.qkv_proj(x)

        # 2) 切分得到 q,k,v：每个都是 (B, T, C)
        # 为什么：后续注意力需要 QK^T，V 用于加权求和
        q, k, v = qkv.chunk(3, dim=-1)

        # 3) reshape + transpose 拆成多头
        # (B, T, C) -> (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        # 为什么要 transpose：让 head 维度更靠前，便于 batch matmul
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 4) 计算注意力打分：att = QK^T / sqrt(d_k)
        # q: (B, h, T, d)   k: (B, h, T, d)
        # k.transpose(-2, -1): (B, h, d, T)
        # 结果 att: (B, h, T, T)
        # 为什么除 sqrt(d_k)：避免点积随维度增大而方差变大，softmax 饱和导致梯度差
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 5) 加因果 mask：禁止看未来 token
        # mask True 的地方是未来位置，我们填 -inf
        causal_mask = self._get_causal_mask(T, x.device)
        att = att.masked_fill(causal_mask, float("-inf"))

        # 6) softmax 得到注意力权重（沿最后一维：对所有 key 归一化）
        # 结果仍是 (B, h, T, T)，每行和为 1
        att = F.softmax(att, dim=-1)

        # 7) 注意力 dropout（训练时正则化）
        att = self.attn_dropout(att)

        # 8) 加权求和：y = att @ V
        # att: (B, h, T, T)   v: (B, h, T, d)  -> y: (B, h, T, d)
        y = att @ v

        # 9) 合并多头：transpose 回来再 contiguous 再 view
        # (B, h, T, d) -> (B, T, h, d) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 10) 输出投影 + residual dropout
        # 为什么需要 out_proj：多头拼接后再线性变换，允许 heads 之间信息重新混合
        y = self.resid_dropout(self.out_proj(y))
        return y


class FeedForward(nn.Module):
    """
    最小可用 FFN：两层线性 + 激活
    GPT 系常用 GELU；这里使用 GELU 贴近主流
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 第一层扩大维度：让每个 token 的表示进入更高维空间做非线性变换
        x = self.fc1(x)
        # 2) GELU：平滑非线性，实践上比 ReLU 更适合 Transformer
        x = F.gelu(x)
        # 3) 第二层投回 d_model：回到残差主干维度
        x = self.fc2(x)
        # 4) dropout：正则
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    一个标准的 Decoder-only block（Pre-LN 版本）
    Pre-LN：先 LayerNorm 再子层；训练更稳定，深层更容易收敛
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 注意力子层（Pre-LN）
        # 为什么 residual：保持信息主干，缓解梯度问题，让深层训练可行
        x = x + self.attn(self.ln1(x))
        # 2) FFN 子层（Pre-LN）
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """
    最小可用 GPT 风格语言模型
    """
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # token embedding：把 token id 映射为 d_model 向量
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # position embedding：让模型知道顺序（最小实现用可学习位置向量）
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.drop = nn.Dropout(dropout)

        # 堆叠 Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 最后再 LN 一次（GPT 常见做法）
        self.ln_f = nn.LayerNorm(d_model)

        # 语言模型头：投影到 vocab logits
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 常见的 weight tying：lm_head 与 tok_emb 共享权重（可选但常用）
        # 为什么：减少参数量，且通常略提升效果
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx: (B, T) token ids
        targets: (B, T) 训练用的目标 token（通常是 idx 右移一位）
        返回:
          logits: (B, T, vocab)
          loss:   scalar（如果提供 targets）
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, "序列长度超过 max_seq_len"

        # 1) token embedding： (B,T) -> (B,T,C)
        tok = self.tok_emb(idx)

        # 2) position ids：0..T-1
        # 为什么要显式构造：pos_emb 需要索引；同时支持不同长度 T
        pos_ids = torch.arange(T, device=idx.device)

        # 3) position embedding： (T,C) broadcast 到 (B,T,C)
        pos = self.pos_emb(pos_ids)

        # 4) 相加得到输入表示（token 语义 + 位置信息）
        x = self.drop(tok + pos)

        # 5) 过 N 个 block
        for blk in self.blocks:
            x = blk(x)

        # 6) 最后 LN
        x = self.ln_f(x)

        # 7) 得到 vocab logits： (B,T,C) -> (B,T,vocab)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # 8) 语言模型训练：预测每个位置的下一个 token
            # 交叉熵期望输入形状为 (N, vocab) 和 (N,)
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1)
            )

        return logits, loss


def demo_run():
    torch.manual_seed(0)

    vocab_size = 5000
    max_seq_len = 64
    model = MiniGPT(vocab_size=vocab_size, max_seq_len=max_seq_len)

    B, T = 8, 32
    idx = torch.randint(0, vocab_size, (B, T))

    # 语言模型训练通常 targets = idx 的“下一位”
    # 最小示例：把 targets 设为随机也能跑通；这里做一个右移示例
    targets = torch.roll(idx, shifts=-1, dims=1)

    logits, loss = model(idx, targets)
    loss.backward()

    print("logits:", logits.shape)  # (B, T, vocab)
    print("loss:", float(loss))


if __name__ == "__main__":
    demo_run()
