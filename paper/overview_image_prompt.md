# SpeechMaster Overview Figure Prompt

Create a clean ICASSP-style research system diagram for a speech recognition paper.

Title inside the figure: "SpeechMaster: Complementarity-Aware SSL-ASR Routing".

Use a white background, thin dark-gray strokes, muted blue and green accents, and a professional academic style. Do not use cartoon characters or decorative gradients.

Diagram layout:

1. Left input block: "Waveform x".
2. Main top pipeline from left to right:
   - "Fast SSL-ASR: Wav2Vec2 CTC"
   - outputs "hypothesis + logits"
   - "Fast-branch features: CTC confidence, entropy, duration, word/char rates"
   - "CAR gain router: predict HuBERT edit-count gain"
   - "Budget decision: route top B% predicted gain"
   - "Final transcript"
3. Lower routed branch:
   - from the budget decision down to "Strong SSL-ASR: HuBERT CTC"
   - arrow back up to "Final transcript" labeled "replace high-gain utterances"
4. Separate lower analysis branch from waveform:
   - "HuBERT hidden states"
   - "K-means unit auditor"
   - outputs "token rate, bitrate, codebook size"
5. Add a small side note box near the router:
   - "Training target on dev cache: edit(F(x), y) - edit(S(x), y)"
   - "Inference uses fast-branch features only"

Make the routed ASR core visually grouped with a dashed rectangle labeled "Budgeted ASR core". Make the unit analysis branch grouped with another dashed rectangle labeled "Representation-cost audit". Use arrows with clear labels and keep all text readable at two-column paper width.
