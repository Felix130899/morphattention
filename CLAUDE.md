## Name
masterthesis_fish_VIT

## Purpose
XAI on ViT attention maps for fish morphology (NHM Wien imagery)

## Non-negotiable constraints
- Local-only processing, no cloud/external APIs
- Only Neo + supervisor access; weights never published
- NHM attribution required in public outputs

## Current state
- SAM ViT-L integration: loader + smoke test done, not yet wired into devcontainer
- CUDA/NVML issue was specific to the sandboxed dev session, not this
  machine: on the actual host, `nvidia-smi` works (driver 595.84, RTX 4070
  Ti), `nvidia-container-toolkit` was already installed, and
  `docker-compose.yml` (gitignored, reconstructed this session — see
  vault/thesis-log/decisions/2026-09-19-gitignore-docker-compose.md) builds
  and passes `torch.cuda.is_available() == True` inside the container with
  no changes needed to the file or the Dockerfile. Re-verified from inside
  the rebuilt dev container by loading a real model (`load_sam()` lands on
  `cuda:0`, not just the availability flag).

## Next steps
1. Wire into devcontainer.json postCreateCommand
2. Pin verified deps in requirements.txt

## Detailed log
See vault/thesis-log/ (symlinked, see below)

## Update protocol (read this before editing this file)
- "Current State" and "Next Steps": OVERWRITE, never append.
- Max 5 bullets each. Hitting the cap means something belongs in
  vault/thesis-log/ instead — link it, don't inline it.
- Only include state changes that are true as of THIS session.
- A "Next Steps" item only counts if it's concrete enough to start
  without re-deriving context (not "improve model" — "add SamPredictor
  fallback for tray images with >1 fish per mask").
- Never touch "Purpose" or "Non-negotiable constraints" unless Neo
  explicitly asks.