# Model Comparison on BNCI2014001 Subject 1

| Model | Balanced Accuracy | Stage 1 Binary | Stage 2 MI | Params |
|-------|:----------------:|:--------------:|:----------:|:------:|
| conformer | **0.662** | 0.752 | 0.726 | Heavy |
| **eegnet_attn** | **0.650** | 0.718 | 0.742 | Light |
| eegnet | 0.613 | 0.731 | 0.706 | Lightest |
| deepcnn | 0.555 | 0.770 | 0.622 | Medium |
| deformer | 0.547 | 0.594 | 0.640 | Heavy |
| shallowconvnet | 0.545 | 0.732 | 0.671 | Light |
| dbconformer | 0.527* | — | — | Heaviest |

*\*dbconformer OOM on test set, validation only.*

## Key Findings

1. **Conformer is best**, but EEGNet_Attn (your model) is close behind at 1.2% gap with far fewer parameters
2. **EEGNet_Attn has the best Stage 2 MI accuracy** (0.742) — it's best at distinguishing left/right/feet/tongue
3. **Original EEGNet is a strong baseline** — beats all other CNN variants
4. **Deep CNNs underperform** — deeper isn't better for this dataset
5. **Voting hurts performance** — single window > any voting strategy for these models

## Recommendation

Use **conformer** if you want max accuracy, or **eegnet_attn** for a good balance of accuracy and speed.
