"""Train DA-012 with ASSA replacing the LAF cross branch at P3 and P4."""
from tools.assareplace_laf_runtime import run


results = run(
    "pre-pth/yolov8s-obb_twostream_assareplace_lafcross_p34_refinep34_v1.pt",
    "ASSALAF_ReplaceLAFCross_P34_RefineP34_H2-4_R4-StaticDW3-NoFFN_v1",
)
