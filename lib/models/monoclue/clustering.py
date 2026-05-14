import torch
import torch.nn.functional as F
from torch import nn
import copy
import numpy as np
import os
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from einops import rearrange
from fast_pytorch_kmeans import KMeans
from .memory import SceneMemory
from .ops.modules import MSDeformAttn_Relocalization

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_in, d_model, nhead, dim_feedforward=256, dropout=0.1, activation="gelu", class_num=1):
        super().__init__()

        self.nhead = nhead
        self.linear1 = nn.Linear(d_model, dim_feedforward )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.class_num = class_num
        self.dropout_p = dropout

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.k = nn.Linear(d_in, d_model)
        self.v = nn.Linear(d_in, d_model)
        self.q = nn.Linear(d_in, d_model)
        self.proj = nn.Linear(d_model, d_model)

    def _get_activation_fn(self, activation):
        """Return an activation function given a string"""
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if activation == "glu":
            return F.glu
        raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

    def key_masking(self, token, key):
        key_padding_mask = (key == 0).all(dim=2)
        is_all_zero_batch = key_padding_mask.all(dim=1)
        valid_batch_indices = (~is_all_zero_batch).nonzero(as_tuple=False).squeeze(1)

        if is_all_zero_batch.all():
            return None, None, None, None

        # Remove the batch if all values of C are zero within N of the batch
        valid_key_padding_mask = key_padding_mask[valid_batch_indices]
        valid_token = token[valid_batch_indices]
        valid_key = key[valid_batch_indices]

        return valid_token, valid_key, valid_key_padding_mask, valid_batch_indices

    def forward(self, token, key):

        valid_token, valid_key, valid_key_padding_mask, valid_batch_indices = self.key_masking(token, key)

        if valid_key_padding_mask is None:
            return token

        _, N, C = valid_token.shape

        q = self.q(valid_token)
        k = self.k(valid_key)
        v = self.v(valid_key)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.nhead)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.nhead)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.nhead)

        # Invert the key padding.
        attn_mask = ~valid_key_padding_mask  # (B, S)
        attn_mask = attn_mask[:, None, None, :]  # (B,1,1,S)
        attn_mask = attn_mask.expand(-1, self.nhead, N, -1)

        src2 = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale = (C // self.nhead) ** -0.5, dropout_p = self.dropout_p)
        src2 = rearrange(src2, 'b h n d -> b n (h d)')
        src2 = self.proj(src2)
        q = rearrange(q, 'b h n d -> b n (h d)')

        src = q + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)

        # skip
        src_full = torch.zeros_like(token).to(token.device)  # (B, S, C)
        src_full[valid_batch_indices] = src

        token = token + src_full

        return token

class VisualEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn_Relocalization(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def get_reference_points(self, spatial_shapes, valid_ratios, corr_map, device):
        reference_points_list = []
        delta_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)

            corr = corr_map[lvl]
            corr = corr.masked_fill(corr < 0.5, float('-inf'))
            is_all_inf = torch.isinf(corr)
            if is_all_inf.all():
                corr_softmax = torch.zeros_like(corr)
            else:
                corr_softmax = F.softmax(corr, dim=1)
            # center = corr @ ref  # (B, 2)
            center = torch.bmm(corr_softmax.transpose(1, 2).contiguous(), ref)
            delta = (center - ref).clamp(-0.5, 0.5) * corr_softmax

            delta_list.append(delta)
            reference_points_list.append(ref)

        reference_points = torch.cat(reference_points_list, 1)
        delta_list = torch.cat(delta_list, 1)
        reference_points = reference_points + delta_list # offset initialize
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]

        return reference_points

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, deform_info, corr_map):

        spatial_shapes = deform_info[0]
        level_start_index = deform_info[1]
        valid_ratios = deform_info[2]
        padding_mask = deform_info[3]
        pos_embeding = deform_info[4]

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, corr_map, src.device)

        correlation = torch.cat(corr_map, dim=1)
        query = self.with_pos_embed(src, pos_embeding)

        # self attention
        src2 = self.self_attn(torch.cat([query, correlation],dim=-1), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.forward_ffn(src)

        return src

class QueryInitializer(nn.Module):
    def __init__(self, hidden_dim, cfg=None):
        super().__init__()
        cfg = cfg or {}
        self.hidden_dim = hidden_dim
        self.n_levels = 4
        self.mem_slots = 1  # class num
        self.use_foreground = cfg.get('use_foreground', True)
        self.use_background = cfg.get('use_background', True)
        self.use_scene_memory = cfg.get('use_scene_memory', True)
        self.use_relocalization = cfg.get('use_relocalization', True)
        self.fg_num_clusters = cfg.get('fg_num_clusters', 10)
        self.bg_num_clusters = cfg.get('bg_num_clusters', 3)

        self.global_mem = nn.ModuleList([SceneMemory(self.mem_slots, self.hidden_dim) for _ in range(3)])
        self.token_attention = TransformerEncoderLayer(self.hidden_dim, self.hidden_dim,8, 384)
        self.expand_feature = VisualEncoderLayer(self.hidden_dim, self.hidden_dim, 0.1, "relu", self.n_levels, 8, 4)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn_Relocalization):
                m._reset_parameters()

    def split_feature(self, x, spatial_shapes):
        out = []
        ptr = 0
        for h, w in spatial_shapes:
            n = h * w
            feat = x[:, ptr:ptr + n]
            out.append(rearrange(feat, 'b (h w) c -> b c h w', h=h, w=w))
            ptr += n

        return out

    def masked_avg(self, feat, mask, eps=1e-6):
        """
        feat : (B,C,H,W)
        mask : (B,K,1,H,W)
        output_size : (P_h, P_w)
        return : (B,K,C,P_h,P_w)  (squeeze→(B,K,C), When P_h=P_w=1)
        """
        B, C, H, W = feat.shape
        K = mask.size(1)
        x = feat.unsqueeze(1) * mask  # (B,K,C,H,W)

        area = mask.sum((-1, -2)).clamp_min(eps)  # (B,K,1)
        pooled = x.flatten(-2).sum(-1) / area  # (B,K,C)
        return pooled

    def get_feature_clustering_masks(self, x, region, num_clusters=10):
        if num_clusters <= 0:
            B, _, H, W = region.size()
            return torch.zeros(B, 0, H, W, device=x.device, dtype=x.dtype)

        B, _, H, W = region.size()
        masks = []
        max_num_clusters = num_clusters
        for b in range(B):
            batch_slice = x[b]
            region_prob = region[b]

            flattened_prob = region_prob.view(-1)
            indices = torch.where(flattened_prob > 0.5)[0]

            if indices.numel() == 0:
                batch_masks = torch.zeros(max_num_clusters, H, W).to(x.device)
                masks.append(batch_masks.unsqueeze(0))
                continue

            cur_num_clusters = min(num_clusters, indices.numel())

            batch_slice = rearrange(batch_slice, 'c h w -> (h w) c')
            high_prob_features = batch_slice[indices, :]

            kmeans = KMeans(n_clusters=cur_num_clusters, mode='euclidean', max_iter=10, verbose=0)
            labels = kmeans.fit_predict(high_prob_features)
            batch_masks = torch.zeros(max_num_clusters, H * W).to(x.device)
            for i in range(cur_num_clusters):
                mask_indices = indices[labels == i]
                batch_masks[i, mask_indices] = 1.
            batch_masks = rearrange(batch_masks, 'c (h w) -> c h w', h=H, w=W).unsqueeze(0)
            masks.append(batch_masks)

        masks = torch.cat(masks, dim=0)

        return masks

    def get_correlation_map(self, x):

        min_vals = x.min(dim=2, keepdim=True)[0]
        max_vals = x.max(dim=2, keepdim=True)[0]

        denom = (max_vals - min_vals).clamp(min=1e-8)
        corr = (x - min_vals) / denom
        corr = corr ** 2
        corr = corr * 2 - 1
        corr = F.sigmoid(corr)

        corr = torch.max(corr, dim=1, keepdim=True)[0]
        corr = (corr - torch.min(corr)) / (torch.max(corr) - torch.min(corr) + 1e-6)
        corr = rearrange(corr, 'b c n -> b n c')

        return corr

    def split_mask(self, masks):
        B, C, H, W = masks.size()
        if C <= 1:
            bottom = masks.new_zeros(B, 0, H, W)
            top = masks
            return masks, bottom, top

        H_top = H // 4
        top_mask = masks[:, :, :H_top, :]  # (B, C, H_top, W)

        all_cols = torch.arange(C, device=masks.device).unsqueeze(0).repeat(B, 1)

        # Count the number of 1s in the top 1/4 region.
        area_scores = top_mask.sum(dim=(2, 3))  # (B, C)

        # Extract the max area and its index from each batch.
        max_vals, remove_idx = area_scores.max(dim=1)  # (B,), (B,)

        bottom_idx = all_cols != remove_idx.unsqueeze(1)
        bottom_idx = bottom_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # (B, C, H, W)
        bottom = masks[bottom_idx].view(B, C - 1, H, W)

        top_idx = all_cols == remove_idx.unsqueeze(1)
        top_idx = top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        top = masks[top_idx].view(B, 1, H, W)

        has_positive = max_vals > 0  # (B,)

        filtered_masks = masks.clone()  # (B, C, H, W)

        for b in range(B):
            if has_positive[b]:
                filtered_masks[b, remove_idx[b]] = 0  # 0으로 마스킹

        return filtered_masks, bottom, top


    @staticmethod
    def _debug_tensor(x):
        return x.detach().float().cpu()

    def forward(self, x, region, query, deform_info, return_debug=False):

        B, _, _ = x.shape

        spatial_shapes = deform_info[0]
        debug = {
            'spatial_shapes': self._debug_tensor(spatial_shapes),
            'fg_cluster_masks': [],
            'bg_cluster_masks': [],
            'sky_cluster_masks': [],
            'corr_maps': [],
        } if return_debug else None

        # ------------------- multi-scale feature ------------------------------
        s8, s16, s32, s64 = [f for f in self.split_feature(x, spatial_shapes)]

        multi_feat = [s8, s16, s32]
        multi_scale = [1, 2, 4]

        corr_map = []
        back_clustering = []
        scene_clustering = []
        local_clustering = []
        for i, (feat, s, m) in enumerate(zip(multi_feat, multi_scale, region)):

            # foreground
            if self.use_foreground:
                kmeans_mask = self.get_feature_clustering_masks(feat, m, self.fg_num_clusters)
                if return_debug:
                    debug['fg_cluster_masks'].append(self._debug_tensor(kmeans_mask))
                kmeans_mask = kmeans_mask.unsqueeze(2)
                local_cluster_feature = self.masked_avg(feat, kmeans_mask)
                local_clustering.append(local_cluster_feature)
            else:
                local_cluster_feature = None

            # background
            if self.use_background:
                back = 1 - m
                back_mask = self.get_feature_clustering_masks(feat, back, num_clusters=self.bg_num_clusters)
                back_mask, bottom, top = self.split_mask(back_mask)  # remove sky
                if return_debug:
                    debug['bg_cluster_masks'].append(self._debug_tensor(bottom))
                    debug['sky_cluster_masks'].append(self._debug_tensor(top))
                if bottom.size(1) > 0:
                    bottom = bottom.unsqueeze(2)
                    back_cluster_feature = self.masked_avg(feat, bottom)
                    back_clustering.append(back_cluster_feature)

            # scene_memory
            if self.use_scene_memory and local_cluster_feature is not None:
                scene_cluster_feature = self.global_mem[i](local_cluster_feature)
                scene_clustering.append(scene_cluster_feature)

            # re-localization
            if self.use_relocalization and local_cluster_feature is not None:
                _, _, nh, nw = feat.shape
                feat_flatten = rearrange(feat, 'b c h w -> b (h w) c')
                corr = torch.bmm(local_cluster_feature, feat_flatten.transpose(1, 2).contiguous())  # b n (h w)
                corr = self.get_correlation_map(corr)
                corr_map.append(corr)
                if return_debug:
                    debug['corr_maps'].append(self._debug_tensor(corr))

                if i == len(multi_feat) - 1:
                    corr = rearrange(corr, 'b (h w) c -> b c h w', h=nh, w=nw)
                    corr = F.interpolate(corr, size=(nh // 2, nw // 2), mode='nearest')
                    corr = rearrange(corr, 'b c h w -> b (h w) c')
                    corr_map.append(corr)
                    if return_debug:
                        debug['corr_maps'].append(self._debug_tensor(corr))

        # Relocalization
        if self.use_relocalization and len(corr_map) == self.n_levels:
            re_localization = self.expand_feature(x, deform_info, corr_map)
        else:
            re_localization = x

        # clustering features
        cluster_parts = []
        if local_clustering:
            cluster_parts.append(torch.cat(local_clustering, dim=1))
        if back_clustering:
            cluster_parts.append(torch.cat(back_clustering, dim=1))
        if scene_clustering:
            cluster_parts.append(torch.cat(scene_clustering, dim=1))

        if cluster_parts:
            cluster = torch.cat(cluster_parts, dim=1)  # (B, N+M, C)
            query = self.token_attention(query, cluster)


        if return_debug:
            return query, re_localization, debug

        return query, re_localization


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
