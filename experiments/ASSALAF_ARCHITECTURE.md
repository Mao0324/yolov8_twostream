# ASSA-LAF v1: Align -> Fuse -> Refine

The combined model uses one module for each distinct failure mode instead of
stacking every successful experiment block.

- **P3/P4 alignment:** `PartialChannelASSAFusion` exchanges information on C/4
  channels with static DW3 projections and no FFN. It addresses RGB/thermal
  correspondence while keeping most channels on an untouched residual path.
- **P3/P4 fusion and feedback:** `LAFMergeFeedback2D` forms the neck lateral and
  returns only the learned fusion correction to both modality backbones.
- **P3/P4 fused refinement:** `ZeroInitResidualRefine2D` repairs detail after
  fusion, where DA-012 showed the strongest useful placement.
- **P5:** original LAF only. Coarse semantic features do not receive another
  ASSA or refine block.

`StaticMAA2D`, full-channel `PaperLAF`, target-saliency supervision, and P5 ASSA
are intentionally omitted: they overlap with alignment/fusion attention or did
not improve the current DarkAct accuracy-speed frontier.

The migration starts from DA-012. New ASSA output projections and refine gates
are zero initialized, so the initial network exactly retains the DarkAct source
function while allowing alignment to enter training gradually.

## Cross-replacement follow-ups

The original align-then-fuse run reached 0.706 test mAP50-95 versus 0.707 for
DA-012. Its launcher is archived at
`experiments/assalaf/archive/alignfuse_p34_train.py` and is not a current
experiment entrypoint. The follow-ups therefore remove the duplicated cross
interaction:

- `ALF-001`: replace the LAF cross branch with partial ASSA at P3 and P4;
  retain the complete DA-012 LAF at P5.
- `ALF-002`: replace the LAF cross branch only at P3; retain the complete
  DA-012 LAF at P4 and P5.

At every replaced scale, LAF retains only its global channel and local spatial
reliability gates. ASSA is the sole cross-modal interaction and contributes its
two directional residuals to the fused lateral. The original feedback formula
and P3/P4 fused refine blocks remain unchanged.
