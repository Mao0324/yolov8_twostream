# Report design

- Audience: technical
- Delivery: portable self-contained HTML
- Reading path: answer first, experiment ranking, class-level gains, learned mechanism, architecture interpretation, scope/method, limitations, next ablations, open questions
- Visual 1: full-width horizontal bar chart of test mAP50-95 delta versus baseline for the three DarkAct variants
- Visual 2: full-width grouped bar chart of per-class test mAP50-95 for baseline and V2, because exact class-level breadth is central to the winning-architecture claim
- Tables: exact experiment/complexity audit table and V2/V3 stage-mechanism tables
- Palette: blue and gold roots plus neutrals; no red/green semantics
- Responsive behavior: single-column reading flow, full-width evidence blocks, native portable reader, semantic table fallbacks
- Source strategy: calculated claims resolve to `source_notes.md` and `source_data.sql`; raw experiment artifacts, YAMLs, modules, and `analysis.py` are separately declared

