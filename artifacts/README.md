
# Model and Retrieval Artifacts

This directory stores the trained model checkpoint and visual retrieval
artifacts required by the Medical AI Chatbot MVP.

## Required files

| File | Description |
|---|---|
| `densenet121_chexpert.pth` | Trained DenseNet-121 classification checkpoint |
| `chexpert_visual.index` | FAISS visual retrieval index |
| `chexpert_embeddings.npy` | Precomputed image embeddings |
| `chexpert_metadata.csv` | Reference image metadata and labels |

## How to obtain the files

The files are shared separately through the team's shared Google Drive
folder because they may be too large for GitHub.

Download the required files and place them directly inside this directory.

## Important

- Do not commit model checkpoints, datasets, or generated index files.
- Ensure the model architecture and preprocessing match the feature
  extraction implementation.
- The retrieval index was generated using the training reference subset.
- The validation subset should remain separate from the reference index.

## Note

This project is an experimental educational prototype and is not
clinically validated. All model outputs require qualified clinical review.
