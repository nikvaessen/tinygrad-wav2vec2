"""
A tinygrad implementation of the wav2vec 2.0 network.
"""
from typing import Optional

from tinygrad import Tensor, nn


########################################################################################
# The feature network transforms raw audio X to local speech representations Z


def grad_scale(x: Tensor, scale: float) -> Tensor:
    inverse_scale = 1 - scale if -1 <= scale <= 1 else -1 / scale
    return (x * scale) + (x.detach() * inverse_scale)


class FeatureNetwork:
    def __init__(self, grad_scaling: float = 1 / 10):
        self.c1 = nn.Conv1d(1, 512, 10, 5, 1)
        self.c1_norm = nn.GroupNorm(512, 512)

        self.c2 = nn.Conv1d(512, 512, 3, 2, 1)
        self.c3 = nn.Conv1d(512, 512, 3, 2, 1)
        self.c4 = nn.Conv1d(512, 512, 3, 2, 1)
        self.c5 = nn.Conv1d(512, 512, 3, 2, 1)

        self.c6 = nn.Conv1d(512, 512, 2, 2)
        self.c7 = nn.Conv1d(512, 512, 2, 2)

        self.grad_scale = grad_scaling

    def __call__(self, x: Tensor) -> Tensor:
        # X has shape [BATCH_SIZE, 1, SEQ_LENGTH]

        h = x
        h = self.c1_norm(self.c1(h)).gelu()
        h = self.c2(h).gelu()
        h = self.c3(h).gelu()
        h = self.c4(h).gelu()
        h = self.c5(h).gelu()
        h = self.c6(h).gelu()
        h = self.c7(h).gelu()

        # we return shape [BATCH_SIZE, SEQ_LENGTH/320, 512]
        return grad_scale(h, self.grad_scale).transpose(1, 2)


########################################################################################
# The context network transforms local representations Z to contextual representations C


class SelfAttention:
    def __init__(self, num_dim: int, num_heads: int, dropout_prob: float):
        super().__init__()
        assert num_dim % num_heads == 0

        # the key, value and query projection as a single (batched) layer
        self.attention_projection = nn.Linear(num_dim, num_dim * 3)

        # logic for multi-head attention
        self.num_dim = num_dim
        self.num_heads = num_heads

        self.head_dim = num_dim // num_heads

        # the projection after the concatenation scaled dot-product attention output
        self.out_projection = nn.Linear(num_dim, num_dim)

        # regularization
        self.dropout_prob = dropout_prob

    def __call__(self, x: Tensor, attention_mask: Optional[Tensor] = None):
        # x has shape [BATCH_SIZE, SEQUENCE_LENGTH, NUM_DIM]
        # attention mask has shape [BATCH_SIZE, NUM_HEADS, SEQ_LENGTH, SEQ_LENGTH]
        bs, seq_length, num_dim = x.shape

        # we first compute the key/query/value output
        k, q, v = self.attention_projection(x).split(self.num_dim, dim=2)

        # we then split the output into heads with shape
        # [BATCH_SIZE, SEQUENCE_LENGTH, NUM_HEADS, NUM_DIM_HEAD]
        k = k.view(bs, seq_length, self.num_heads, self.head_dim)
        q = q.view(bs, seq_length, self.num_heads, self.head_dim)
        v = v.view(bs, seq_length, self.num_heads, self.head_dim)

        # We transpose the key/query/value to shape
        # [BATCH_SIZE, NUM_HEADS, SEQUENCE_LENGTH, NUM_DIM_HEAD]
        # so that we easily compute operations for each head separately
        k, q, v = k.transpose(1, 2), q.transpose(1, 2), v.transpose(1, 2)

        # now we can apply self-attention for each head such that
        # `y ~= softmax( (QK^T) / sqrt(num_dim) )* V`
        # note the attention mask ensures feature vectors from padded time steps are
        # ignored as their attention score is set to -inf
        y = q.scaled_dot_product_attention(
            key=k,
            value=v,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob,
            is_causal=False,
        )

        # we concatenate the heads, so that we go back to the original shape
        y = y.transpose(1, 2).contiguous().view(bs, seq_length, num_dim)

        # we apply the final projection and dropout
        y = self.out_projection(y).dropout(self.dropout_prob)

        return y


class FeedForwardNetwork:
    def __init__(self, num_dim: int, hidden_dim: int, dropout_prob: float):
        self.fc1 = nn.Linear(num_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_dim)
        self.dropout_prob = dropout_prob

    def __call__(self, x: Tensor):
        # x has shape [BATCH_SIZE, SEQUENCE_LENGTH, NUM_DIM]
        x = self.fc1(x).gelu()
        x = self.fc2(x).dropout(self.dropout_prob)

        return x


class TransformerEncoderLayer:
    def __init__(
        self, num_dim: int, num_dim_ffn: int, num_heads: int, dropout_prob: float
    ):
        self.attention = SelfAttention(num_dim, num_heads, dropout_prob)
        self.norm_att = nn.LayerNorm(num_dim)

        self.ffn = FeedForwardNetwork(num_dim, num_dim_ffn, dropout_prob)
        self.norm_ffn = nn.LayerNorm(num_dim)

    def __call__(self, x: Tensor, attention_mask: Tensor):
        # x has shape [BATCH_SIZE, SEQUENCE_LENGTH, NUM_DIM]
        # mask has shape [BATCH_SIZE, NUM_HEADS, SEQUENCE_LENGTH, SEQUENCE_LENGTH]
        x = self.norm_att(x + self.attention(x, attention_mask))
        x = self.norm_ffn(x + self.ffn(x))

        return x


class ContextNetwork:
    def __init__(self):
        # projection layer
        self.proj_norm = nn.LayerNorm(512)
        self.proj_linear = nn.Linear(512, 768)

        # positional embedding
        self.rpe_conv = nn.Conv1d(768, 768, 128, 1, 64, 1, 16)
        self.rpe_norm = nn.LayerNorm(768)

        # transformer
        self.encoder_layers = [
            TransformerEncoderLayer(768, 768 * 4, 12, 0.1) for _ in range(12)
        ]

    def __call__(self, x: Tensor):
        # projection
        x = self.proj_linear(self.proj_norm(x))

        # mask
        ...

        # relative positional embedding
        rpe = self.rpe_norm(self.rpe_conv(x).gelu())
        x += rpe

        # context representations
        att_mask = ...
        for layer in self.encoder_layers:
            if x.training and Tensor.rand((1,)).item() < 0.1:
                continue
            x = layer(x, att_mask)

        return x


########################################################################################
# The quantization module computes discrete representations Q from Z

pass


########################################################################################
# pre-training loss functions

pass

########################################################################################
# pre-training network

pass

########################################################################################
# fine-tuning network

pass

########################################################################################
# debugging code


def main():
    cnn = FeatureNetwork()
    wav = Tensor.rand((1, 1, 16_000))

    opt = nn.optim.AdamW(nn.state.get_parameters(cnn), lr=1e-7)

    # step
    Tensor.training = True
    for i in range(1000):
        opt.zero_grad()
        features = cnn(wav)
        loss = features.sum().abs()
        loss.backward()
        opt.step()
        print(loss.numpy())


if __name__ == "__main__":
    main()
