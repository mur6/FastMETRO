# ----------------------------------------------------------------------------------------------
# FastMETRO Official Code
# Copyright (c) POSTECH Algorithmic Machine Intelligence Lab. (P-AMI Lab.) All Rights Reserved
# Licensed under the MIT license.
# ----------------------------------------------------------------------------------------------

"""
FastMETRO model.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .position_encoding import build_position_encoding
from .smpl_param_regressor import build_smpl_parameter_regressor
from .transformer import build_transformer
from src.handinfo.ring.helper import RING_1_INDEX, RING_2_INDEX, WRIST_INDEX


class MLP(nn.Module):
    def __init__(self, input_size=128 * 2, hidden_size1=256 * 4, dropout=0.1, output_size=3):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size1, output_size)
        # self.fc1 = nn.Linear(input_size, hidden_size1)
        # self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        # self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return out


class MLP_3_Layer(nn.Module):
    def __init__(self, input_size=195 * 3, output_size=1):
        super(MLP_3_Layer, self).__init__()
        dropout_prob = 0.5
        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(512, 64)
        self.relu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x


class OnlyRadiusModel(nn.Module):
    def __init__(self, fastmetro_model, *, net_for_radius=None):
        super().__init__()
        self.fastmetro_model = fastmetro_model
        # self.mlp_for_radius = MLP(input_size=195 * 3, hidden_size1=256, dropout=0.6, output_size=1)
        if net_for_radius is None:
            self.net_for_radius = STN3d()
        else:
            self.net_for_radius = net_for_radius

    def forward(self, images, mano_model, *, output_minimum=False):
        (
            pred_cam,
            pred_3d_joints,
            pred_3d_vertices_coarse,
            pred_3d_vertices_fine,
            cam_features,
            enc_img_features,
            jv_features,
        ) = self.fastmetro_model(images)
        pred_3d_joints_from_mano = mano_model.get_3d_joints(pred_3d_vertices_fine)
        pred_3d_joints_from_mano_wrist = pred_3d_joints_from_mano[:, WRIST_INDEX, :]
        pred_3d_vertices_fine = pred_3d_vertices_fine - pred_3d_joints_from_mano_wrist[:, None, :]
        pred_3d_joints = pred_3d_joints - pred_3d_joints_from_mano_wrist[:, None, :]
        ring1_point = pred_3d_joints[:, 13, :]
        ring2_point = pred_3d_joints[:, 14, :]
        plane_normal = ring2_point - ring1_point  # (batch X 3)
        plane_origin = (ring1_point + ring2_point) / 2  # (batch X 3)
        ######### 半径のみ推論
        # batch_size = pred_3d_vertices_coarse.shape[0]
        # x = pred_3d_vertices_coarse.contiguous().view(batch_size, -1)
        pred_3d_vertices_coarse = torch.transpose(pred_3d_vertices_coarse, 2, 1)
        # print(f"pred_3d_vertices_coarse: {pred_3d_vertices_coarse.shape}")
        radius = self.net_for_radius(pred_3d_vertices_coarse)
        # print(f"radius: {radius.shape}")
        if output_minimum:
            return plane_origin, plane_normal, radius
        else:
            return (
                plane_origin,
                plane_normal,
                radius,
                pred_3d_joints,
                pred_3d_vertices_fine,
            )


class SimpleCustomModel(nn.Module):
    def __init__(self, fastmetro_model):
        super().__init__()
        self.fastmetro_model = fastmetro_model
        self.mlp_for_pca_mean = MLP()
        self.mlp_for_normal_v = MLP()

    def forward(self, images):
        cam_features, _, jv_features = self.fastmetro_model(images, output_minimum=True)
        joint_features = jv_features[RING_1_INDEX : RING_2_INDEX + 1, :, :]
        # x = torch.cat((cam_features, joint_features), 0).transpose(0, 1)
        x = joint_features.transpose(0, 1)
        batch_size = x.shape[0]
        x = x.contiguous().view(batch_size, -1)
        # print(f"x: {x.shape}")
        return self.mlp_for_pca_mean(x), self.mlp_for_normal_v(x), None


class DecWide128Model(nn.Module):
    def __init__(self, args, num_joints=21, num_vertices=195):
        super().__init__()
        self.args = args
        self.num_joints = num_joints
        self.num_vertices = num_vertices
        self.num_ring_infos = 3
        assert "FastMETRO-L" in args.model_name
        num_enc_layers = 3
        num_dec_layers = 3
        # configurations for the first transformer
        self.transformer_config_3 = {
            "model_dim": 128,
            "dropout": args.transformer_dropout,
            "nhead": args.transformer_nhead,
            "feedforward_dim": 512,
            "num_enc_layers": num_enc_layers,
            "num_dec_layers": num_dec_layers,
            "pos_type": args.pos_type,
        }
        # build transformers
        self.transformer_3_decoder = build_transformer(self.transformer_config_3).decoder
        # token embeddings
        self.ring_token_embed = nn.Embedding(
            self.num_ring_infos, self.transformer_config_3["model_dim"]
        )
        # # positional encodings
        self.position_encoding_3 = build_position_encoding(
            pos_type=self.transformer_config_3["pos_type"],
            hidden_dim=self.transformer_config_3["model_dim"],
        )
        # estimators
        # self.use_features_num = 5
        in_features = self.transformer_config_3["model_dim"]  # * self.use_features_num
        self.ring_center_regressor = nn.Linear(in_features, 3)
        self.ring_normal_regressor = nn.Linear(in_features, 3)
        self.radius_regressor = nn.Linear(in_features, 1)

    def _do_decode(self, hw, bs, device, enc_img_features, jv_tokens, pos_embed):
        # hw, bs = img_features.shape  # (height * width), batch_size, feature_dim
        mask = torch.zeros(
            (bs, hw), dtype=torch.bool, device=device
        )  # batch_size X (height * width)
        # Transformer Decoder
        zero_tgt = torch.zeros_like(
            jv_tokens
        )  # (num_joints + num_vertices) X batch_size X feature_dim
        decoder = self.transformer_3_decoder
        attention_mask = None
        jv_features = decoder(
            jv_tokens,
            enc_img_features,
            tgt_mask=attention_mask,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=zero_tgt,
        )  # (num_joints + num_vertices) X batch_size X feature_dim
        return jv_features

    def forward(self, cam_features_2, enc_img_features_2, jv_features_2):
        device = cam_features_2.device
        batch_size = cam_features_2.size(1)
        # fastmetro:cam_features_3: torch.Size([1, 1, 128])
        # fastmetro:enc_img_features_3: torch.Size([49, 1, 128])
        # fastmetro:jv_features_3: torch.Size([216, 1, 128])
        h, w = 7, 7
        # positional encodings
        pos_enc_3 = (
            self.position_encoding_3(batch_size, h, w, device).flatten(2).permute(2, 0, 1)
        )  # 49 X batch_size X 128
        # print(f"forward:[prev]jv_features_2: {jv_features_2.shape}")
        # print(f"forward:[prev]ring_token_embed.weight: {self.ring_token_embed.weight.shape}")
        r_tokens = self.ring_token_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        # print(f"forward:r_tokens: {r_tokens.shape}")
        r_tokens_and_jv = torch.cat([r_tokens, jv_features_2], dim=0)
        # print(f"forward:r_tokens_and_jv: {r_tokens_and_jv.shape}")
        jv_features_final = self._do_decode(
            h * w, batch_size, device, enc_img_features_2, r_tokens_and_jv, pos_enc_3
        )
        # print(f"jv_features_final: {jv_features_final.shape}")
        # cam_features, enc_img_features, jv_features = self.transformer_3(
        #     enc_img_features_2, cam_features_2, jv_features_2, pos_enc_3
        # )
        # pred_cam = self.cam_predictor(cam_features_2).view(batch_size, 3)  # batch_size X 3
        center_features = jv_features_final[[0], :, :].transpose(0, 1)
        # center_features = center_features.contiguous().view(-1, 128 * 3)
        center = self.ring_center_regressor(center_features)
        normal_v_features = jv_features_final[[1], :, :].transpose(0, 1)
        # normal_v_features = normal_v_features.contiguous().view(-1, 128 * 3)
        normal_v = self.ring_normal_regressor(normal_v_features)
        radius_features = jv_features_final[[2], :, :].transpose(0, 1)
        radius = self.radius_regressor(radius_features)
        return (
            center.squeeze(1),
            normal_v.squeeze(1),
            radius.squeeze(1),
        )


class T3EncDecModel(nn.Module):
    def __init__(self, args, num_joints=21, num_vertices=195):
        super().__init__()
        self.args = args
        self.num_joints = num_joints
        self.num_vertices = num_vertices
        self.num_ring_infos = 3
        assert "FastMETRO-L" in args.model_name
        num_enc_layers = 2
        num_dec_layers = 2
        # configurations for the first transformer
        self.transformer_config_3 = {
            "model_dim": 32,
            "dropout": args.transformer_dropout,
            "nhead": args.transformer_nhead,
            "feedforward_dim": 32 * 4,
            "num_enc_layers": num_enc_layers,
            "num_dec_layers": num_dec_layers,
            "pos_type": args.pos_type,
        }

        self.dim_reduce_enc_cam = nn.Linear(128, self.transformer_config_3["model_dim"])
        self.dim_reduce_enc_img = nn.Linear(128, self.transformer_config_3["model_dim"])
        self.dim_reduce_dec = nn.Linear(128, self.transformer_config_3["model_dim"])

        # build transformers
        self.transformer_3 = build_transformer(self.transformer_config_3)
        # token embeddings
        self.ring_token_embed = nn.Embedding(
            self.num_ring_infos, self.transformer_config_3["model_dim"]
        )
        # # positional encodings
        self.position_encoding_3 = build_position_encoding(
            pos_type=self.transformer_config_3["pos_type"],
            hidden_dim=self.transformer_config_3["model_dim"],
        )
        # estimators
        # self.use_features_num = 5
        in_features = self.transformer_config_3["model_dim"]  # * self.use_features_num
        self.ring_center_regressor = nn.Linear(in_features, 3)
        self.ring_normal_regressor = nn.Linear(in_features, 3)
        self.radius_regressor = nn.Linear(in_features, 1)

    def forward(self, cam_features_2, enc_img_features_2, jv_features_2):
        device = cam_features_2.device
        batch_size = cam_features_2.size(1)

        # progressive dimensionality reduction
        reduced_cam_features_2 = self.dim_reduce_enc_cam(cam_features_2)  # 1 X batch_size X 32
        reduced_enc_img_features_2 = self.dim_reduce_enc_img(
            enc_img_features_2
        )  # 49 X batch_size X 32
        jv_features_2 = self.dim_reduce_dec(jv_features_2)
        # (num_joints + num_vertices) X batch_size X 32

        # fastmetro:cam_features_3: torch.Size([1, 1, 128])
        # fastmetro:enc_img_features_3: torch.Size([49, 1, 128])
        # fastmetro:jv_features_3: torch.Size([216, 1, 128])
        h, w = 7, 7
        # positional encodings
        pos_enc_3 = (
            self.position_encoding_3(batch_size, h, w, device).flatten(2).permute(2, 0, 1)
        )  # 49 X batch_size X 128
        # print(f"forward:[prev]jv_features_2: {jv_features_2.shape}")
        # print(f"forward:[prev]ring_token_embed.weight: {self.ring_token_embed.weight.shape}")
        r_tokens = self.ring_token_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        # print(f"forward:r_tokens: {r_tokens.shape}")
        r_tokens_and_jv = torch.cat([r_tokens, jv_features_2], dim=0)
        # print(f"forward:r_tokens_and_jv: {r_tokens_and_jv.shape}")

        # cam_features_2, enc_img_features_2, jv_features_2 = self.transformer_2(
        #     reduced_enc_img_features_1,
        #     reduced_cam_features_1,
        #     reduced_jv_features_1,
        #     pos_enc_2,
        #     attention_mask=attention_mask,
        # )
        _, _, jv_features_final = self.transformer_3(
            reduced_enc_img_features_2,
            reduced_cam_features_2,
            r_tokens_and_jv,
            pos_enc_3,
            attention_mask=None,
        )
        # pred_cam = self.cam_predictor(cam_features_2).view(batch_size, 3)  # batch_size X 3
        center_features = jv_features_final[[0], :, :].transpose(0, 1)
        # center_features = center_features.contiguous().view(-1, 128 * 3)
        center = self.ring_center_regressor(center_features)
        normal_v_features = jv_features_final[[1], :, :].transpose(0, 1)
        # normal_v_features = normal_v_features.contiguous().view(-1, 128 * 3)
        normal_v = self.ring_normal_regressor(normal_v_features)
        radius_features = jv_features_final[[2], :, :].transpose(0, 1)
        radius = self.radius_regressor(radius_features)
        return (
            center.squeeze(1),
            normal_v.squeeze(1),
            radius.squeeze(1),
        )


if __name__ == "__main__":
    model = STNkd()
    # m.eval()
    x = torch.randn(8, 195, 3)
    output = model(x)
    print(output.shape)
    # main(args.resume_dir, args.input_filename)
