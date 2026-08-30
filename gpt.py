import torch
import torch.nn as nn
import config

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self,x):
        return self.embedding(x)
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.w_up = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_down(self.dropout(torch.nn.functional.silu(self.w_gate(x)) * self.w_up(x)))

class RoPE(nn.Module):
    def __init__(self, d_k: int, seq_len: int):
        super().__init__()
        self.d_k = d_k
        self.seq_len = seq_len

        pos = torch.arange(seq_len).unsqueeze(1)
        pairs = d_k // 2
        i = torch.arange(pairs)
        freq = 10000 ** ((-2*i) / d_k)

        angles = pos * freq

        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        self.register_buffer("cos_angles", cos_angles)
        self.register_buffer("sin_angles", sin_angles)

    def forward(self, x):
        batch, heads, seq_len, d_k = x.shape
        x_pairs = x.view(
            batch,
            heads,
            seq_len,
            d_k // 2,
            2
        )

        cos_squeeze = self.cos_angles[:seq_len].unsqueeze(0).unsqueeze(0)
        sin_squeeze = self.sin_angles[:seq_len].unsqueeze(0).unsqueeze(0)

        x0 = x_pairs[..., 0]
        x1 = x_pairs[..., 1]

        new_x0 = (cos_squeeze * x0) - (sin_squeeze * x1)
        new_x1 = (cos_squeeze * x1) + (sin_squeeze * x0)

        combine = torch.stack([new_x0, new_x1], dim=-1)

        return combine.reshape(batch, heads, seq_len, d_k)
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float, seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.rope = RoPE(self.d_k, seq_len)

    @staticmethod
    def attention(query, key, value, mask, dropout: float):
        return nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            mask,
            dropout,
            is_causal=False
        )
        #d_k = query.shape[-1]

        #attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        #if mask is not None:
        #    attention_scores.masked_fill_(mask == 0, float("-inf"))
        #attention_scores = attention_scores.softmax(dim = -1)
        #if dropout is not None:
        #    attention_scores = dropout(attention_scores)
#
        #return (attention_scores @ value), attention_scores
    
    def forward(self, q,k,v, mask, dropout):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        query = self.rope(query)
        key = self.rope(key)

        x = MultiHeadAttentionBlock.attention(query, key, value, mask, dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 10**-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self,x):
        sqr = x**2
        rms = (sqr.mean(dim=-1, keepdim=True)  + self.eps)**0.5
        return (x / rms) * self.alpha
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float, d_model):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float, d_model):
        super().__init__()
        self.self_attention_block =  self_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = dropout
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout, d_model) for _ in range(2)])

    def forward(self, x, mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, mask, self.dropout))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model):
        super().__init__()
        self.layers = layers
        self.norm = RMSNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias = False)

    def forward(self, x):
        return self.proj(x)
    
class Transformer(nn.Module):

    def __init__(self, decoder: Decoder, embed: InputEmbeddings, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.decoder = decoder
        self.embedding = embed
        self.projection_layer = projection_layer

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.decoder(x, mask)
        logits = self.projection_layer(x)

        return logits
    
def build_transformer(vocab_size: int, seq_len: int, d_model: int=config.d_model, N: int=config.layers, h: int=config.heads, dropout: float=config.dropout, d_ff: int=config.d_ff) -> Transformer:
    embed = InputEmbeddings(d_model, vocab_size)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout, seq_len)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, feed_forward_block, dropout, d_model)
        decoder_blocks.append(decoder_block)
    
    decoder = Decoder(nn.ModuleList(decoder_blocks), d_model)
    
    projection_layer = ProjectionLayer(d_model, vocab_size)

    projection_layer.proj.weight = embed.embedding.weight
    
    transformer = Transformer(decoder, embed, projection_layer)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

def create_causal_mask(size):
    mask = torch.tril(torch.ones(size, size))

    return mask.bool().unsqueeze(0).unsqueeze(0)
    
if __name__ == "__main__":

    d_model = 512
    vocab_size = 32000
    heads = 8
    batch_size = 2
    seq_len = 32
    d_k = 64
    epochs = 300

    x = torch.randint(
        0,
        vocab_size,
        (batch_size,seq_len)
    )
    targets = torch.randint(
            0,
            vocab_size,
            (batch_size, seq_len)
        )
    targets = targets.reshape(-1)

    tran = build_transformer(vocab_size,seq_len)
    mask = create_causal_mask(seq_len)
    total_loss = 0
    
    optimizer = torch.optim.AdamW(
            tran.parameters(),
            1e-3
        )
    criterion = nn.CrossEntropyLoss()
    print(int(0.1))
    for i in range(epochs):
        tran.train()
        optimizer.zero_grad()
        y = tran(x,mask)
        logits = y.reshape(
            -1,
            y.size(-1)
        )
        loss = criterion(logits,targets)
        loss.backward()

        optimizer.step()
        total_loss = loss.item()
        if (i%10 == 0):
            print(f"Epoch: {i}  //  Loss: {total_loss}")

