"""Train DA-012 with ASSA replacing the LAF cross branch only at P3."""
from tools.assareplace_laf_runtime import run


results = run(
    "pre-pth/yolov8s-obb_twostream_assareplace_lafcross_p3_refinep34_v1.pt",
    "ASSALAF_ReplaceLAFCross_P3_RefineP34_H2_R4-StaticDW3-NoFFN_v1",
)
