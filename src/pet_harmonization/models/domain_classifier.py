import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
from src.pet_harmonization.models.base import UnetResBlock, BasicBlock, BasicDown

class DomainClassifier(nn.Module):
    def __init__(
        self,
        in_ch: int = 5,
        num_domains: int = 2,
        spatial_dims: int = 2,
        hid_chs: List[int] = [64, 128, 256, 512],
        kernel_sizes: List[int] = [3, 3, 3, 3],
        strides: List[int] = [1, 2, 2, 2],
        act_name: Tuple[str, Dict] = ('swish', {}),
        norm_name: Tuple[str, Dict] = ('group', {'num_groups': 32, 'affine': True}),
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__()
        
        self.depth = len(hid_chs)
        self.num_domains = num_domains
        
        # --- 1. In-Convolution (comme dans vos autres modèles) ---
        self.in_conv = BasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_ch,
            out_channels=hid_chs[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0],
        )

        # --- 2. Encoder Path (Extraction de caractéristiques) ---
        # Utilise vos UnetResBlock pour la cohérence
        encoder_layers = []
        for i in range(1, self.depth):
            # Bloc de convolution résiduel
            encoder_layers.append(
                UnetResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=hid_chs[i-1],
                    out_channels=hid_chs[i],
                    kernel_size=kernel_sizes[i],
                    stride=1,
                    norm_name=norm_name,
                    act_name=act_name,
                    dropout=dropout
                )
            )
            # Descente spatiale (Downsampling)
            if strides[i] > 1:
                encoder_layers.append(
                    BasicDown(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i],
                        out_channels=hid_chs[i],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i]
                    )
                )
        
        self.encoder = nn.Sequential(*encoder_layers)

        # --- 3. Classification Head ---
        # Global Average Pooling pour être indépendant de la taille du patch en entrée
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1) if spatial_dims == 2 else nn.AdaptiveAvgPool3d(1)
        
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hid_chs[-1], hid_chs[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hid_chs[-1] // 2, num_domains)
        )

    def forward(self, x):
        # x: [B, 5, 64, 64]
        h = self.in_conv(x)
        features = self.encoder(h)
        
        # Pooling pour obtenir un vecteur latent global
        latent = self.global_avg_pool(features)
        
        # Logits de sortie pour la CrossEntropyLoss
        logits = self.classifier_head(latent)
        
        return logits

    def get_latent_features(self, x):
        h = self.in_conv(x)
        return self.encoder(h)
    

class MultiLevelDomainClassifier(nn.Module):
    def __init__(
        self,
        feature_channels: List[int],
        num_domains: int = 3,
        spatial_dims: int = 2,
        dropout: float = 0.2,
        num_conv_per_level: Optional[List[int]] = None,
    ):
        super().__init__()
        self.num_levels = len(feature_channels)

        # Règle automatique si non spécifié
        if num_conv_per_level is None:
            num_conv_per_level = []
            for i in range(self.num_levels):
                if i == 0:
                    num_conv_per_level.append(2)   # niveau bas : 2 conv
                elif i == self.num_levels - 1:
                    num_conv_per_level.append(0)   # latent : pas de conv
                else:
                    num_conv_per_level.append(1)   # niveaux intermédiaires : 1 conv

        assert len(num_conv_per_level) == self.num_levels, (
            f"num_conv_per_level should have {self.num_levels} elements, got {len(num_conv_per_level)}"
        )

        classifiers = []
        for ch, n_conv in zip(feature_channels, num_conv_per_level):
            classifiers.append(
                self._make_level_classifier(ch, num_domains, n_conv, spatial_dims, dropout)
            )
        self.classifiers = nn.ModuleList(classifiers)

    def _make_level_classifier(
        self,
        ch: int,
        num_domains: int,
        num_conv: int,
        spatial_dims: int,
        dropout: float,
    ) -> nn.Sequential:
        Conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        BN   = nn.BatchNorm2d if spatial_dims == 2 else nn.BatchNorm3d
        Pool = nn.AdaptiveAvgPool2d(1) if spatial_dims == 2 else nn.AdaptiveAvgPool3d(1)

        layers: List[nn.Module] = []

        # Couches convolutives (same shape)
        for _ in range(num_conv):
            layers += [
                Conv(ch, ch, kernel_size=3, padding=1, bias=False),
                BN(ch),
                nn.ReLU(inplace=True),
            ]

        # Pooling + tête MLP
        hidden = max(ch // 4, 64)
        layers += [
            Pool,
            nn.Flatten(),
            nn.Linear(ch, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_domains),
        ]

        return nn.Sequential(*layers)

    def forward(self, encoder_features: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(encoder_features) == self.num_levels, (
            f"{self.num_levels} levels expected, received: {len(encoder_features)}"
        )
        return [clf(feat) for clf, feat in zip(self.classifiers, encoder_features)]
    

class DeepMultiLevelDomainClassifier(nn.Module):
    def __init__(
        self,
        feature_channels: List[int],
        num_domains: int = 3,
        spatial_dims: int = 2,
        hid_chs: List[int] = [64, 128, 256, 512],
        kernel_sizes: List[int] = [3, 3, 3, 3],
        strides: List[int] = [1, 2, 2, 2],
        act_name: Tuple[str, Dict] = ('swish', {}),
        norm_name: Tuple[str, Dict] = ('group', {'num_groups': 32, 'affine': True}),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_levels = len(feature_channels)

        # Pour N niveaux : depths = [N-1, N-2, ..., 1, 0]
        depths = list(range(self.num_levels - 1, -1, -1))

        Pool = nn.AdaptiveAvgPool2d(1) if spatial_dims == 2 else nn.AdaptiveAvgPool3d(1)

        self.encoders = nn.ModuleList()   # BasicBlock + UnetResBlock + BasicDown
        self.heads = nn.ModuleList()      # Pool + Flatten + MLP

        for level_idx, (in_ch, depth) in enumerate(zip(feature_channels, depths)):

            # ----------------------------------------------------------
            # Encodeur (vide pour le latent)
            # ----------------------------------------------------------
            if depth == 0:
                # Cas latent : aucun bloc, on va directement au pooling
                self.encoders.append(nn.Identity())
                out_ch = in_ch

            else:
                # InConv : BasicBlock(in_ch → hid_chs[0])
                layers: List[nn.Module] = [
                    BasicBlock(
                        spatial_dims=spatial_dims,
                        in_channels=in_ch,
                        out_channels=hid_chs[0],
                        kernel_size=kernel_sizes[0],
                        stride=strides[0],
                    )
                ]

                # Succession de UnetResBlock + BasicDown sur depth niveaux
                # On prend les depth premiers éléments de hid_chs (après le 0)
                for i in range(1, depth + 1):
                    ch_in  = hid_chs[i - 1]
                    ch_out = hid_chs[min(i, len(hid_chs) - 1)]

                    layers.append(
                        UnetResBlock(
                            spatial_dims=spatial_dims,
                            in_channels=ch_in,
                            out_channels=ch_out,
                            kernel_size=kernel_sizes[min(i, len(kernel_sizes) - 1)],
                            stride=1,
                            norm_name=norm_name,
                            act_name=act_name,
                            dropout=dropout,
                        )
                    )

                    if strides[min(i, len(strides) - 1)] > 1:
                        layers.append(
                            BasicDown(
                                spatial_dims=spatial_dims,
                                in_channels=ch_out,
                                out_channels=ch_out,
                                kernel_size=kernel_sizes[min(i, len(kernel_sizes) - 1)],
                                stride=strides[min(i, len(strides) - 1)],
                            )
                        )

                self.encoders.append(nn.Sequential(*layers))
                out_ch = hid_chs[min(depth, len(hid_chs) - 1)]

            # ----------------------------------------------------------
            # Tête de classification (identique pour tous les niveaux)
            # ----------------------------------------------------------
            hidden = max(out_ch // 2, 64)
            self.heads.append(
                nn.Sequential(
                    Pool,
                    nn.Flatten(),
                    nn.Linear(out_ch, hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, num_domains),
                )
            )

    def forward(self, encoder_features: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(encoder_features) == self.num_levels, (
            f"{self.num_levels} levels expected, received: {len(encoder_features)}"
        )
        return [
            head(enc(feat))
            for enc, head, feat in zip(self.encoders, self.heads, encoder_features)
        ]