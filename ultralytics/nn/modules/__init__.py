# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .block import (
    C1,
    C2,
    C3,
    C3TR,
    DFL,
    SPP,
    SPPELAN,
    SPPF,
    ADown,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C3Ghost,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Proto,
    RepC3,
    RepNCSPELAN4,
    ResNetLayer,
    Silence,
    Concat2,
    S2Attention,
    ADD,
    SimAM,
    ShuffleAttention,
    GAM_Attention,
    CBAM2,
    CoordAtt,
    ECA,
    NAM,
    SEAttention,
    SKAttention,
    GLF,
    GLCBAM,
    GCBAM,
    SACBAM,
    MdC2f,
    CDC2f,
    C2f_Shufflenet,
    C2f_Invo,
    C2f_PKIModule,
    CSFM,
    FEM,
    C2f_FEM,
    C2f_PPA,
    C2f_Faster,
    C2f_RG,
    Fusion,
    Concat3,
    RIFusion
)
from .assa_fusion import ASSAFusion
from .assa_partial_channel_fusion import PartialChannelASSAFusion
from .assa_fusion_static_noffn import ASSAFusionStaticNoFFN
from .assa_laf_fusion import (
    ASSALAFMergeFeedback2D,
    ASSAReplacedLAFMerge2D,
    ASSAReplacedLAFMergeFeedback2D,
)
from .darkact_maalaf import LAFMerge2D, MAA2D
from .darkact_maalaf_v2 import LAFMergeFeedback2D, StaticMAA2D, ZeroCenteredStaticMAA2D
from .darkact_maalaf_v3 import PaperLAFMergeFeedback2D
from .darkact_disagreement_fusion import (
    DisagreementLAFMergeFeedback2D,
    SemanticDisagreementLAFMergeFeedback2D,
)
from .darkact_target_saliency import StaticMAAContext2D, TargetSaliencyPaperLAFMergeFeedback2D
from .darkact_target_saliency_fp32safe import StaticMAAContext2DFP32Safe
from .darkact_target_saliency_stable_attention import (
    StaticMAAContext2DL2Temp,
    StaticMAAContext2DSqrtHW,
)
from .ft_fusion import FTCrossMerge
from .proto_hg_fusion import ProtoHypergraphFusion
from .p2det_prompt_gder import (
    P2_PROMPT_MODULES,
    P2_SECOND_GEN_PROMPT_MODULES,
    P2DualPromptGDERMergeFeedback2D,
    P2DualPromptIdentityGDERMergeFeedback2D,
    P2DualPromptLAFMergeFeedback2D,
    P2DualPromptRGBGlobalIdentityGDERFactorizedMergeFeedback2D,
    P2DualPromptRGBGlobalIdentityGDERMergeFeedback2D,
    P2DualPromptRGBGlobalLAFMergeFeedback2D,
    P2DualSpatialLAFMergeFeedback2D,
    P2IRPromptLAFMergeFeedback2D,
    P2IRPromptLAFMergeFeedbackNoStaticMAA2D,
    P2IRPromptAsymIdentityGDERMergeFeedback2D,
    P2IRSpatialRGBGlobalLAFMergeFeedback2D,
)
from .zero_init_refine import ZeroInitResidualRefine2D
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention,
)
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment, WorldDetect
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "WorldDetect",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "Silence",
    "ASSAFusion",
    "PartialChannelASSAFusion",
    "ASSAFusionStaticNoFFN",
    "ASSALAFMergeFeedback2D",
    "ASSAReplacedLAFMerge2D",
    "ASSAReplacedLAFMergeFeedback2D",
    "MAA2D",
    "LAFMerge2D",
    "StaticMAA2D",
    "ZeroCenteredStaticMAA2D",
    "LAFMergeFeedback2D",
    "DisagreementLAFMergeFeedback2D",
    "SemanticDisagreementLAFMergeFeedback2D",
    "PaperLAFMergeFeedback2D",
    "StaticMAAContext2D",
    "StaticMAAContext2DFP32Safe",
    "StaticMAAContext2DSqrtHW",
    "StaticMAAContext2DL2Temp",
    "TargetSaliencyPaperLAFMergeFeedback2D",
    "FTCrossMerge",
    "ProtoHypergraphFusion",
    "P2IRPromptLAFMergeFeedback2D",
    "P2IRPromptLAFMergeFeedbackNoStaticMAA2D",
    "P2IRPromptAsymIdentityGDERMergeFeedback2D",
    "P2DualPromptLAFMergeFeedback2D",
    "P2DualPromptGDERMergeFeedback2D",
    "P2_PROMPT_MODULES",
    "P2IRSpatialRGBGlobalLAFMergeFeedback2D",
    "P2DualSpatialLAFMergeFeedback2D",
    "P2DualPromptIdentityGDERMergeFeedback2D",
    "P2DualPromptRGBGlobalLAFMergeFeedback2D",
    "P2DualPromptRGBGlobalIdentityGDERMergeFeedback2D",
    "P2DualPromptRGBGlobalIdentityGDERFactorizedMergeFeedback2D",
    "P2_SECOND_GEN_PROMPT_MODULES",
    "ZeroInitResidualRefine2D",
)
