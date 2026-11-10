# Evaluation

We evaluate 15 products across 90 frames. All four configurations use the same reference images and editing instructions. The rewriter comparison changes only the prompt supplied to the same RL-adapted editor.

| Configuration | MUSIQ ↑ | Product ↑ | Detail ↑ | Seg CLIP-I ↑ | Seg DINO-I ↑ |
|---|---:|---:|---:|---:|---:|
| Base | 66.5818 | 3.2778 | 3.2889 | 0.7487 | 0.4310 |
| SFT | 67.4980 | 2.8111 | 2.8889 | 0.7413 | 0.3833 |
| SFT + RL | 66.3715 | 3.3778 | 3.3556 | 0.7624 | 0.4419 |
| SFT + RL + Rewriter | **67.5194** | **3.6667** | **3.5333** | **0.7724** | **0.4778** |

MUSIQ measures whole-image quality. InternVL3.5-8B-Instruct rates product identity and detail preservation on separate five-point scales. These three metrics use all 90 frames per configuration.

Seg CLIP-I and Seg DINO-I compare product regions extracted with Grounding DINO and SAM 2. Both metrics use the same 62 frames for which the original instruction calls for product-related content and region extraction succeeds in all four configurations.

## Edit-intent preservation

Both rewriter versions are scored on the same 90 frames from the 15 products. InternVL3.5-8B-Instruct measures how well the rewritten prompt and generated image satisfy requirements from the original instruction. Scores are mean weighted checklist coverage on a 0–100 scale.

| Rewriter | Text intent ↑ | Image completion ↑ |
|---|---:|---:|
| Without task-specific training | 43.21 | 50.76 |
| Trained | **85.78** | **88.79** |
