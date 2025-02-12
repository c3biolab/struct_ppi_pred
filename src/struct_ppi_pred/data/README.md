# Data Module (`src/data`)

This module is responsible for handling the data preprocessing and preparation for the Protein-Protein Interaction (PPI) prediction project. Since this project utilizes a custom dataset, this guide provides instructions on how to prepare and integrate your own dataset for use with the codebase.

## Dataset Requirements

Your dataset must meet the following specifications:

1. **Split Files**:

   - Provide split files (`train.csv`, `val.csv`, `test.csv`) in the `data/` directory.

   - Each file must have three columns:
     - `P1`: UniProt ID of the first protein.
     - `P2`: UniProt ID of the second protein.
     - `Label`: Binary label indicating interaction (`1` for interacting proteins, `0` for non-interacting proteins).

   Example:
   ```csv
   P1,P2,Label
   P12345,P67890,1
   Q23456,Q78901,0
   ```

2. **Protein Structure Data**:

   - Download protein structure files from the [AlphaFold Database](https://alphafold.ebi.ac.uk/) for each protein listed in your split files.

   - Follow the preprocessing steps outlined in the [MAPE-PPI framework](https://github.com/LirongWu/MAPE-PPI) to generate structural features for each protein.

## Folder Structure

Module assumes the following structure for your dataset:

```
data/
├── train.csv          # Training split file
├── val.csv            # Validation split file
├── test.csv           # Test split file
├── processed_data/    # Preprocessed feature files
│   ├── <UniProtID>_r_contacts.npy
│   ├── <UniProtID>_k_contacts.npy
│   └── <UniProtID>_nodes.pt
```

### Key Files in `processed_data/`:
- `<UniProtID>_r_contacts.npy`: Contact features (radius-based) for the protein.
- `<UniProtID>_k_contacts.npy`: Contact features (k-nearest neighbors) for the protein.
- `<UniProtID>_nodes.pt`: Node features for the protein.

Refer to the MAPE-PPI [GitHub repository](https://github.com/LirongWu/MAPE-PPI) and their paper [Wu et al., 2022](https://arxiv.org/abs/2402.14391) for detailed preprocessing steps and file generation.
