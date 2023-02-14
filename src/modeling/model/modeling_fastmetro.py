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
from torch import nn

from .position_encoding import build_position_encoding
from .smpl_param_regressor import build_smpl_parameter_regressor
from .transformer import build_transformer


class FastMETRO_Original_Hand_Network(nn.Module):
    """FastMETRO for 3D hand mesh reconstruction from a single RGB image"""

    def __init__(self, args, backbone, mesh_sampler, num_joints=21, num_vertices=195):
        """
        Parameters:
            - args: Arguments
            - backbone: CNN Backbone used to extract image features from the given image
            - mesh_sampler: Mesh Sampler used in the coarse-to-fine mesh upsampling
            - num_joints: The number of joint tokens used in the transformer decoder
            - num_vertices: The number of vertex tokens used in the transformer decoder
        """
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.mesh_sampler = mesh_sampler
        self.num_joints = num_joints
        self.num_vertices = num_vertices

        # the number of transformer layers
        if "FastMETRO-S" in args.model_name:
            num_enc_layers = 1
            num_dec_layers = 1
        elif "FastMETRO-M" in args.model_name:
            num_enc_layers = 2
            num_dec_layers = 2
        elif "FastMETRO-L" in args.model_name:
            num_enc_layers = 3
            num_dec_layers = 3
        else:
            assert False, "The model name is not valid"

        # configurations for the first transformer
        self.transformer_config_1 = {
            "model_dim": args.model_dim_1,
            "dropout": args.transformer_dropout,
            "nhead": args.transformer_nhead,
            "feedforward_dim": args.feedforward_dim_1,
            "num_enc_layers": num_enc_layers,
            "num_dec_layers": num_dec_layers,
            "pos_type": args.pos_type,
        }
        # configurations for the second transformer
        self.transformer_config_2 = {
            "model_dim": args.model_dim_2,
            "dropout": args.transformer_dropout,
            "nhead": args.transformer_nhead,
            "feedforward_dim": args.feedforward_dim_2,
            "num_enc_layers": num_enc_layers,
            "num_dec_layers": num_dec_layers,
            "pos_type": args.pos_type,
        }
        # build transformers
        self.transformer_1 = build_transformer(self.transformer_config_1)
        self.transformer_2 = build_transformer(self.transformer_config_2)
        # dimensionality reduction
        self.dim_reduce_enc_cam = nn.Linear(
            self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"]
        )
        self.dim_reduce_enc_img = nn.Linear(
            self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"]
        )
        self.dim_reduce_dec = nn.Linear(
            self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"]
        )

        # token embeddings
        self.cam_token_embed = nn.Embedding(1, self.transformer_config_1["model_dim"])
        self.joint_token_embed = nn.Embedding(
            self.num_joints, self.transformer_config_1["model_dim"]
        )
        self.vertex_token_embed = nn.Embedding(
            self.num_vertices, self.transformer_config_1["model_dim"]
        )
        # positional encodings
        self.position_encoding_1 = build_position_encoding(
            pos_type=self.transformer_config_1["pos_type"],
            hidden_dim=self.transformer_config_1["model_dim"],
        )
        self.position_encoding_2 = build_position_encoding(
            pos_type=self.transformer_config_2["pos_type"],
            hidden_dim=self.transformer_config_2["model_dim"],
        )
        # estimators
        self.xyz_regressor = nn.Linear(self.transformer_config_2["model_dim"], 3)
        self.cam_predictor = nn.Linear(self.transformer_config_2["model_dim"], 3)

        # 1x1 Convolution
        self.conv_1x1 = nn.Conv2d(
            args.conv_1x1_dim, self.transformer_config_1["model_dim"], kernel_size=1
        )

        # attention mask
        zeros_1 = torch.tensor(np.zeros((num_vertices, num_joints)).astype(bool))
        zeros_2 = torch.tensor(np.zeros((num_joints, (num_joints + num_vertices))).astype(bool))
        adjacency_indices = torch.load("./src/modeling/data/mano_195_adjmat_indices.pt")
        adjacency_matrix_value = torch.load("./src/modeling/data/mano_195_adjmat_values.pt")
        adjacency_matrix_size = torch.load("./src/modeling/data/mano_195_adjmat_size.pt")
        adjacency_matrix = torch.sparse_coo_tensor(
            adjacency_indices, adjacency_matrix_value, size=adjacency_matrix_size
        ).to_dense()
        temp_mask_1 = adjacency_matrix == 0
        temp_mask_2 = torch.cat([zeros_1, temp_mask_1], dim=1)
        self.attention_mask = torch.cat([zeros_2, temp_mask_2], dim=0)

    def forward(self, images):
        device = images.device
        batch_size = images.size(0)

        # preparation
        cam_token = self.cam_token_embed.weight.unsqueeze(1).repeat(
            1, batch_size, 1
        )  # 1 X batch_size X 512
        jv_tokens = (
            torch.cat([self.joint_token_embed.weight, self.vertex_token_embed.weight], dim=0)
            .unsqueeze(1)
            .repeat(1, batch_size, 1)
        )  # (num_joints + num_vertices) X batch_size X 512
        attention_mask = self.attention_mask.to(
            device
        )  # (num_joints + num_vertices) X (num_joints + num_vertices)

        # extract image features through a CNN backbone
        img_features = self.backbone(images)  # batch_size X 2048 X 7 X 7
        _, _, h, w = img_features.shape
        img_features = (
            self.conv_1x1(img_features).flatten(2).permute(2, 0, 1)
        )  # 49 X batch_size X 512

        # positional encodings
        pos_enc_1 = (
            self.position_encoding_1(batch_size, h, w, device).flatten(2).permute(2, 0, 1)
        )  # 49 X batch_size X 512
        pos_enc_2 = (
            self.position_encoding_2(batch_size, h, w, device).flatten(2).permute(2, 0, 1)
        )  # 49 X batch_size X 128

        # first transformer encoder-decoder
        cam_features_1, enc_img_features_1, jv_features_1 = self.transformer_1(
            img_features, cam_token, jv_tokens, pos_enc_1, attention_mask=attention_mask
        )

        # progressive dimensionality reduction
        reduced_cam_features_1 = self.dim_reduce_enc_cam(cam_features_1)  # 1 X batch_size X 128
        reduced_enc_img_features_1 = self.dim_reduce_enc_img(
            enc_img_features_1
        )  # 49 X batch_size X 128
        reduced_jv_features_1 = self.dim_reduce_dec(
            jv_features_1
        )  # (num_joints + num_vertices) X batch_size X 128

        # second transformer encoder-decoder
        cam_features_2, _, jv_features_2 = self.transformer_2(
            reduced_enc_img_features_1,
            reduced_cam_features_1,
            reduced_jv_features_1,
            pos_enc_2,
            attention_mask=attention_mask,
        )

        # estimators
        pred_cam = self.cam_predictor(cam_features_2).view(batch_size, 3)  # batch_size X 3
        pred_3d_coordinates = self.xyz_regressor(
            jv_features_2.transpose(0, 1)
        )  # batch_size X (num_joints + num_vertices) X 3
        pred_3d_joints = pred_3d_coordinates[:, : self.num_joints, :]  # batch_size X num_joints X 3
        pred_3d_vertices_coarse = pred_3d_coordinates[
            :, self.num_joints :, :
        ]  # batch_size X num_vertices(coarse) X 3
        pred_3d_vertices_fine = self.mesh_sampler.upsample(
            pred_3d_vertices_coarse
        )  # batch_size X num_vertices(fine) X 3

        # out = {}
        # out["pred_cam"] = pred_cam
        # out["pred_3d_joints"] = pred_3d_joints
        # out["pred_3d_vertices_coarse"] = pred_3d_vertices_coarse
        # out["pred_3d_vertices_fine"] = pred_3d_vertices_fine
        out = (
            pred_cam,
            pred_3d_joints,
            pred_3d_vertices_coarse,
            pred_3d_vertices_fine,
        )
        return out


############################


class FastMETRO_Hand_Network(nn.Module):
    """FastMETRO for 3D hand mesh reconstruction from a single RGB image"""

    def __init__(
        self, args, backbone, mesh_sampler, num_joints=21, num_vertices=195, num_ring_infos=7
    ):
        """
        Parameters:
            - args: Arguments
            - backbone: CNN Backbone used to extract image features from the given image
            - mesh_sampler: Mesh Sampler used in the coarse-to-fine mesh upsampling
            - num_joints: The number of joint tokens used in the transformer decoder
            - num_vertices: The number of vertex tokens used in the transformer decoder
        """
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.mesh_sampler = mesh_sampler
        self.num_joints = num_joints
        self.num_vertices = num_vertices
        self.num_ring_infos = num_ring_infos

        # the number of transformer layers
        if "FastMETRO-S" in args.model_name:
            num_enc_layers = 1
            num_dec_layers = 1
        elif "FastMETRO-M" in args.model_name:
            num_enc_layers = 2
            num_dec_layers = 2
        elif "FastMETRO-L" in args.model_name:
            num_enc_layers = 3
            num_dec_layers = 3
        else:
            assert False, "The model name is not valid"

        # configurations for the first transformer
        self.transformer_config_1 = {
            "model_dim": args.model_dim_1,
            "dropout": args.transformer_dropout,
            "nhead": args.transformer_nhead,
            "feedforward_dim": args.feedforward_dim_1,
            "num_enc_layers": num_enc_layers,
            "num_dec_layers": num_dec_layers,
            "pos_type": args.pos_type,
        }
        # configurations for the second transformer
        self.transformer_config_2 = {
            "model_dim": args.model_dim_2,
            "dropout": args.transformer_dropout,
            "nhead": args.transformer_nhead,
            "feedforward_dim": args.feedforward_dim_2,
            "num_enc_layers": num_enc_layers,
            "num_dec_layers": num_dec_layers,
            "pos_type": args.pos_type,
        }
        # build transformers
        self.transformer_1 = build_transformer(self.transformer_config_1)
        self.transformer_2 = build_transformer(self.transformer_config_2)
        # dimensionality reduction
        self.dim_reduce_enc_cam = nn.Linear(
            self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"]
        )
        self.dim_reduce_enc_img = nn.Linear(
            self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"]
        )
        self.dim_reduce_dec = nn.Linear(
            self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"]
        )

        # token embeddings
        self.cam_token_embed = nn.Embedding(1, self.transformer_config_1["model_dim"])
        self.joint_token_embed = nn.Embedding(
            self.num_joints, self.transformer_config_1["model_dim"]
        )
        self.vertex_token_embed = nn.Embedding(
            self.num_vertices, self.transformer_config_1["model_dim"]
        )
        self.ring_infos_token_embed = nn.Embedding(
            self.num_ring_infos, self.transformer_config_1["model_dim"]
        )
        # positional encodings
        self.position_encoding_1 = build_position_encoding(
            pos_type=self.transformer_config_1["pos_type"],
            hidden_dim=self.transformer_config_1["model_dim"],
        )
        self.position_encoding_2 = build_position_encoding(
            pos_type=self.transformer_config_2["pos_type"],
            hidden_dim=self.transformer_config_2["model_dim"],
        )
        # estimators
        self.xyz_regressor = nn.Linear(self.transformer_config_2["model_dim"], 3)
        self.cam_predictor = nn.Linear(self.transformer_config_2["model_dim"], 3)

        # 1x1 Convolution
        self.conv_1x1 = nn.Conv2d(
            args.conv_1x1_dim, self.transformer_config_1["model_dim"], kernel_size=1
        )

        # attention mask [TODO]
        zeros_1 = torch.tensor(np.zeros((num_vertices, num_joints)).astype(bool))
        zeros_2 = torch.tensor(np.zeros((num_joints, (num_joints + num_vertices))).astype(bool))
        # / attention mask [TODO]
        adjacency_indices = torch.load("./src/modeling/data/mano_195_adjmat_indices.pt")
        adjacency_matrix_value = torch.load("./src/modeling/data/mano_195_adjmat_values.pt")
        adjacency_matrix_size = torch.load("./src/modeling/data/mano_195_adjmat_size.pt")
        adjacency_matrix = torch.sparse_coo_tensor(
            adjacency_indices, adjacency_matrix_value, size=adjacency_matrix_size
        ).to_dense()
        temp_mask_1 = adjacency_matrix == 0
        temp_mask_2 = torch.cat([zeros_1, temp_mask_1], dim=1)
        self.attention_mask = torch.cat([zeros_2, temp_mask_2], dim=0)

    def forward(self, images, *, output_features=False):
        device = images.device
        batch_size = images.size(0)

        # preparation
        cam_token = self.cam_token_embed.weight.unsqueeze(1).repeat(
            1, batch_size, 1
        )  # 1 X batch_size X 512
        jv_tokens = (
            torch.cat([self.joint_token_embed.weight, self.vertex_token_embed.weight], dim=0)
            .unsqueeze(1)
            .repeat(1, batch_size, 1)
        )  # (num_joints + num_vertices) X batch_size X 512
        attention_mask = self.attention_mask.to(
            device
        )  # (num_joints + num_vertices) X (num_joints + num_vertices)

        # extract image features through a CNN backbone
        img_features = self.backbone(images)  # batch_size X 2048 X 7 X 7
        _, _, h, w = img_features.shape
        # print(f"0:img_features: h={h} w={w}")
        img_features = (
            self.conv_1x1(img_features).flatten(2).permute(2, 0, 1)
        )  # 49 X batch_size X 512

        # positional encodings
        pos_enc_1 = (
            self.position_encoding_1(batch_size, h, w, device).flatten(2).permute(2, 0, 1)
        )  # 49 X batch_size X 512
        pos_enc_2 = (
            self.position_encoding_2(batch_size, h, w, device).flatten(2).permute(2, 0, 1)
        )  # 49 X batch_size X 128

        # first transformer encoder-decoder
        # print(f"1:img_features: {img_features.shape}")
        # print(f"1:cam_token: {cam_token.shape}")
        # print(f"1:jv_tokens: {jv_tokens.shape}")
        # print(f"1:pos_enc_1: {pos_enc_1.shape}")
        cam_features_1, enc_img_features_1, jv_features_1 = self.transformer_1(
            img_features, cam_token, jv_tokens, pos_enc_1, attention_mask=attention_mask
        )

        # progressive dimensionality reduction
        reduced_cam_features_1 = self.dim_reduce_enc_cam(cam_features_1)  # 1 X batch_size X 128
        reduced_enc_img_features_1 = self.dim_reduce_enc_img(
            enc_img_features_1
        )  # 49 X batch_size X 128
        reduced_jv_features_1 = self.dim_reduce_dec(
            jv_features_1
        )  # (num_joints + num_vertices) X batch_size X 128

        # second transformer encoder-decoder
        # print(f"2:reduced_enc_img_features_1: {reduced_enc_img_features_1.shape}")
        # print(f"2:reduced_cam_features_1: {reduced_cam_features_1.shape}")
        # print(f"2:reduced_jv_features_1: {reduced_jv_features_1.shape}")
        # print(f"2:pos_enc_2: {pos_enc_2.shape}")
        cam_features_2, enc_img_features_2, jv_features_2 = self.transformer_2(
            reduced_enc_img_features_1,
            reduced_cam_features_1,
            reduced_jv_features_1,
            pos_enc_2,
            attention_mask=attention_mask,
        )

        # estimators
        pred_cam = self.cam_predictor(cam_features_2).view(batch_size, 3)  # batch_size X 3
        pred_3d_coordinates = self.xyz_regressor(
            jv_features_2.transpose(0, 1)
        )  # batch_size X (num_joints + num_vertices) X 3
        pred_3d_joints = pred_3d_coordinates[:, : self.num_joints, :]  # batch_size X num_joints X 3
        pred_3d_vertices_coarse = pred_3d_coordinates[
            :, self.num_joints :, :
        ]  # batch_size X num_vertices(coarse) X 3
        pred_3d_vertices_fine = self.mesh_sampler.upsample(
            pred_3d_vertices_coarse
        )  # batch_size X num_vertices(fine) X 3

        if output_features:
            out = (
                cam_features_2,
                enc_img_features_2,
                jv_features_2,
            )
        else:
            # out = {}
            # out["pred_cam"] = pred_cam
            # out["pred_3d_joints"] = pred_3d_joints
            # out["pred_3d_vertices_coarse"] = pred_3d_vertices_coarse
            # out["pred_3d_vertices_fine"] = pred_3d_vertices_fine
            out = (
                pred_cam,
                pred_3d_joints,
                pred_3d_vertices_coarse,
                pred_3d_vertices_fine,
                cam_features_2,
                enc_img_features_2,
                jv_features_2,
            )
        return out


class MyModel(nn.Module):
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

        # self.dim_reduce_enc_cam = nn.Linear(128, self.transformer_config_3["model_dim"])
        self.dim_reduce_enc_img = nn.Linear(128, self.transformer_config_3["model_dim"])
        self.dim_reduce_dec = nn.Linear(128, self.transformer_config_3["model_dim"])

        # build transformers
        self.transformer_3 = build_transformer(self.transformer_config_3).decoder
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
        attention_mask = None
        # jv_features = self.transformer_3(
        #     jv_tokens,
        #     enc_img_features,
        #     tgt_mask=attention_mask,
        #     memory_key_padding_mask=mask,
        #     pos=pos_embed,
        #     query_pos=zero_tgt,
        # )  # (num_joints + num_vertices) X batch_size X feature_dim
        _cam_features, _enc_img_features, jv_features = self.transformer_3(
            jv_tokens,
            enc_img_features,
            tgt_mask=attention_mask,
        )
        return jv_features

    def forward(self, cam_features_2, enc_img_features_2, jv_features_2):
        device = cam_features_2.device
        batch_size = cam_features_2.size(1)

        # progressive dimensionality reduction
        # reduced_cam_features_2 = self.dim_reduce_enc_cam(cam_features_2)  # 1 X batch_size X 32
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
        jv_features_final = self._do_decode(
            h * w, batch_size, device, reduced_enc_img_features_2, r_tokens_and_jv, pos_enc_3
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
