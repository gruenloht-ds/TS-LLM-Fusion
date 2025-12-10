import torch
from torch import nn
from timm.models.vision_transformer import trunc_normal_

from .patch import PatchEmbedding
from .mask import MaskGenerator
from .positional_encoding import PositionalEncoding
from .TransformerLayer import TransformerLayers
import torch.nn.functional as F

def unshuffle(shuffled_tokens):
    dic = {}
    for k, v in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index

class ConvLinear(nn.Module):
    def __init__(self, input_dim=0, hidden_dim = 768, num_linear_layers=5):
        super(ConvLinear, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        if num_linear_layers > 1:
            self.linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_linear_layers)])

    def forward(self, x):
        x = self.linear1(x)
        x = nn.GELU()(x)
        if hasattr(self, 'linears'):
            for i in range(len(self.linears)):
                x = self.linears[i](x)
                x = nn.GELU()(x)
        return x

class PatchToEmbeddings(nn.Module):
    def __init__(self, embed_dim=768, patch_dim=96, dropout=0.1):
        super(PatchToEmbeddings, self).__init__()
        self.embed_projection = nn.Linear(embed_dim, patch_dim)
        self.patch_projection = nn.Linear(patch_dim, patch_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, patches, embeddings):
        """
        Args:
            patches: [B, V, P, D] -> [6, 307, 336, 96]
                B is batch size
                V is variate
                P is patch
                D is patch
            embeddings: [100,768]

        Returns:
            patch: [B, V, P]
        """

        B, V, P, D = patches.shape
        E = embeddings.shape[0]

        # Step 1: embeddings into patches
        projected_embeddings = self.embed_projection(embeddings) # [100, 96]
        projected_embeddings = F.normalize(projected_embeddings, dim=-1)

        patches = self.patch_projection(patches) # [B, V, P, 96]
        patches = F.normalize(patches, dim=-1)

        # Step 2:
        # patches reshape [B*V*P, D], projected embeddings
        patches_reshaped = patches.view(-1, D) # [B*V*P, 96]
        similarity = torch.matmul(patches_reshaped, projected_embeddings.T)

        # Step 3:
        # patch
        patch_classes = torch.argmax(similarity, dim=-1)

        # Step 4:
        # Reshape [B, V, P]
        patch_classes = patch_classes.view(B,V,P)
        similarity = similarity.view(B, V, P, E)

        return patch_classes, similarity

class TSFormer(nn.Module):
    """An efficient unsupervised pre-trained model
    for Time Series based on transformer blocks """

    def __init__(self, patch_size, in_channel, embed_dim, num_heads, mlp_ratio, dropout, num_token, mask_ratio,
                 encoder_depth, decoder_depth, stage='pre-train'):
        super(TSFormer,self).__init__()
        assert stage in ['pre-train', 'forecasting'], "Error: incorrect stage type"
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_token = num_token
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.stage = stage
        self.mlp_ratio = mlp_ratio

        self.selected_feature = 0
        # PatchToEmbedding
        #if embed_path != None:
        self.patch_2_embedding = PatchToEmbeddings()
        self.out_patch_2_embedding = PatchToEmbeddings()
        #self.word_embeddings = torch.load(embed_path)
        #self.word_embeddings = self.word_embeddings.to("cuda")

        # Normalization Layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # Encoder Specifics
        # Makes patches and embed
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)

        # Positioning Encoding
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout)

        # Masking
        self.mask = MaskGenerator(num_token, mask_ratio)

        # Encoder
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)

        # Decoder Details
        # Transformer Layer
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)
        # Mask Tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        # Decoder
        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)

        # Prediction Layer
        self.output_layer = nn.Linear(embed_dim, patch_size)

        self.initialize_weights()

    def initialize_weights(self):
        # Positional encoding
        nn.init.uniform_(self.positional_encoding.position_embedding, -0.02, 0.02)
        # Mask Token
        trunc_normal_(self.mask_token, std=0.02)

    def encoding(self, long_term_history, mask=True):
        """ Encoding process of TSFormer: patchify, positional encoding, mask, transformer layers

        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS used in TSFormer
                shape [B, N, 1, P * L]
            mask (bool): True in pre-training stage and False in forecasting stage.

        Returns:
            torch.Tensor: hidden states of unmasked tokens
            list: unmasked token index
            list: masked token index
            """

        batch_size, num_nodes, _, _ = long_term_history.shape [6, 307, 1, 4032]

        # Patchify and embed
        patches = self.patch_embedding(long_term_history)
        patches = patches.transpose(-1,-2)

        classes, similarity = self.patch_2_embedding(patches, self.word_embeddings)

        # Positional embeddings
        patches = self.positional_encoding(patches)

        # Mask
        if mask:
            unmasked_token_index, masked_token_index = self.mask()
            encoder_input = patches[:, :, unmasked_token_index, :]
        else:
            unmasked_token_index, masked_token_index = None, None
            encoder_input = patches

        # Encoding
        hidden_states_unmasked = self.encoder(encoder_input)
        hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)

        return hidden_states_unmasked, unmasked_token_index, masked_token_index, classes, similarity

    def decoding(self, hidden_states_unmasked, masked_token_index):
        """Decoding process of TSFormer: encoder 2 decoder layer, add mask tokens, Transformer layers

        Args:
            hidden_states_unmasked (torch.Tensor): hidden states of masked tokens
                shape [B, N, P*(1-r), d]
            masked_token_index (list): masked token index

        Returns:
        torch.Tensor: reconstructed data
            """

        batch_size, num_nodes, _, _ = hidden_states_unmasked.shape

        # Encoder to Decoder Layer
        hidden_states_unmasked = self.enc_2_dec_emb(hidden_states_unmasked)

        # Add Mask Tokens
        hidden_states_masked = self.positional_encoding(
            self.mask_token.expand(batch_size, num_nodes, len(masked_token_index), hidden_states_unmasked.shape[-1]),
            index=masked_token_index
        )

        hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)

        # Decoding
        hidden_states_full = self.decoder(hidden_states_full)
        hidden_states_full = self.decoder_norm(hidden_states_full)

        # Prediction
        reconstruction_full = self.output_layer(hidden_states_full.view(batch_size, num_nodes, -1, self.embed_dim))

        return reconstruction_full

    def get_reconstructed_masked_token(self, reconstruction_full, real_value_full, unmasked_token_index, masked_token_index):
        """Get reconstructed masked tokens and corresponding ground-truth for subsequent loss computing.

        Args:
            reconstruction_full (torch.Tensor): reconstructed full token
            real_value_full (torch.Tensor): ground truth full tokens.
            unmasked_token_index (list): List of indices for unmasked tokens
            masked_token_index (list): List of indices for masked tokens

        Returns:
            torch.Tensor reconstructed masked tokens
            torch.Tensor ground truth mask tokens

        """

        # Get Reconstructed masked tokens
        batch_size, num_nodes, _, _ = reconstruction_full.shape
        reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]
        reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_nodes, -1).transpose(1,2)

        label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :].transpose(1,2)
        label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous()
        label_masked_tokens = label_masked_tokens.view(batch_size, num_nodes, -1).transpose(1,2)

        return reconstruction_masked_tokens, label_masked_tokens

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor=None, batch_seen: int = None, epoch: int = None, **kwargs) -> torch.Tensor:
        """Feed forward of TSFormer.
            TSFormer has two stages: pretraining and forecasting

        Args:
            history_data: Very long-term historical data time series
                shape [B, L * P, N, 1]
            future_data:
            batch_seen:
            epoch:
            **kwargs:

        Returns:
            pre-training:
                torch.Tensor: reconstruction of masked tokens shape [B, L * P * r, N, 1]
                torch.Tensor: Ground truth of masked tokens shape [B, L * P * r, N, 1]
                dict: data for plotting
            forecasting:
                torch.Tensor: Output of TSFormer of encoder with shape [B, N, L, 1]
        """

        # Reshape
        # History_data = history_data.permute(0, 2, 3, 1)
        # Feed Forward

        if self.stage == 'pre-train':
            # Encoding
            # History_data

            hidden_states_unmasked, unmasked_token_index, masked_token_index, classes, _ = self.encoding(history_data)

            # Classes [6, 307, 336]
            # Decoding
            reconstruction_full = self.decoding(hidden_states_unmasked, masked_token_index)

            # For subsequent loss computation
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_token(reconstruction_full, history_data, unmasked_token_index, masked_token_index)

            reconstruction_full = reconstruction_full.reshape(reconstruction_full.shape[0], reconstruction_full.shape[1],
                                                              reconstruction_full.shape[2] * reconstruction_full.shape[3])
            reconstruction_full = reconstruction_full.unsqueeze(-1)
            reconstruction_full = reconstruction_full.permute(0, 1, 3, 2)
            reconstruction_full_emb = self.patch_embedding(reconstruction_full)
            reconstruction_full_emb = reconstruction_full_emb.transpose(-1, -2)

            _, classes_pred = self.out_patch_2_embedding(reconstruction_full_emb, self.word_embeddings)

            return reconstruction_masked_tokens, label_masked_tokens, classes, classes_pred
        else:
            hidden_states_full, _, _, classes, _ = self.encoding(history_data, mask=False)

            return hidden_states_full, classes



