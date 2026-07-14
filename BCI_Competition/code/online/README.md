# Legacy online inference example

This directory describes the original 128 Hz compatibility demo, not the formal causal online evaluation. It expects a legacy checkpoint, 8--30 Hz input resampled to 128 Hz, 22 channels by 256 samples, and a 0.5-second step. The resampling path may access future samples and must not be reported as strict causal evaluation.

Use `BCI_Competition/docs/evaluation_protocol.md` and the 250 Hz runners under `code/eval/` for the formal workflow.
