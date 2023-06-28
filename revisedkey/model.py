import torch
import torch.nn as nn
import torch.nn.functional as F
from revisedkey.utils import load_pretrain_embeddings

class Reviser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        input_size = args.dimension * 4
        self.key_map = nn.Sequential(
            nn.Linear(input_size, args.ffn_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.ffn_size, args.dimension))

        # load pre-trained embeddings
        source_embeddings = load_pretrain_embeddings(args.source_model)
        target_embeddings = load_pretrain_embeddings(args.target_model)

        vocab_size = source_embeddings.size(0)
        self.source_embed = nn.Embedding(vocab_size, args.dimension)
        self.target_embed = nn.Embedding(vocab_size, args.dimension)
        self.set_pretrain_embeddings(source_embeddings, target_embeddings)


    def set_pretrain_embeddings(self, source_embeddings, target_embeddings):
        self.source_embed.weight = nn.Parameter(source_embeddings)
        self.target_embed.weight = nn.Parameter(target_embeddings)
        self.source_embed.weight.requires_grad = False
        self.target_embed.weight.requires_grad = False


    def key_forward(self, source_hidden, target_hidden, token, **kwargs):
        source_token_embed = self.source_embed(token)
        target_token_embed = self.target_embed(token)

        increase_hidden = self.key_map(torch.cat([
                source_hidden, 
                target_hidden, 
                source_token_embed, 
                target_token_embed
            ], dim=-1))
        hidden = source_hidden + increase_hidden

        return hidden, increase_hidden
    

    def forward(self, source_hidden, target_hidden, token):
        return self.key_forward(source_hidden, target_hidden, token)

    def query_forward(self, hidden):
        return hidden
    
    def transform(self, source_hidden, target_hidden, token):
        return self.key_forward(source_hidden, target_hidden, token)
