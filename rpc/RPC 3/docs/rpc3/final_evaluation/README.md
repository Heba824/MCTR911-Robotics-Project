# RPC 3 presentation package

- [Final evaluation report](FINAL_EVALUATION.md) · methods, results, evidence and figure interpretations
- [Demo guide](DEMO_GUIDE.md) · reproducible WSL build, simulator, controller and validation commands
- [Oral defense notes](DEFENSE_NOTES.md) · eighteen questions and concise answers
- [Figures](figures/) · 21 PNGs at 240 DPI
- [Tables](tables/) · tracking, timing, safety, per-joint RMSE, live trials and counts, GUI terminal sample, dynamics, singularity sweep and damping law
- [Evidence manifest](evidence/sources.json) · original paths relative to `docs/rpc3/`. The evidence folder includes the selected raw files plus the offline recheck results.

The report uses the matched test pipe for its comparison figures, separate WSL live headless trial tables, a standalone GUI dataset, and labeled numerical singularity calculations. Earlier setup failures and test attempts remain in the workspace. The bundled live evidence is the explicitly selected successful Stage 3/4 verification set.

Keep this directory intact when adding it to the project ZIP. The plotting script reads the bundled inputs and imports the actual robot FK and FLC from the project. Use the demo guide to source Jazzy and expose the source package. The generator runs without launching the simulator, and the bundled PNGs and tables can be opened directly.
