# CycleGAN: Unpaired Image-to-Image Translation

An implementation of Cycle-Consistent Generative Adversarial Networks (CycleGAN) built using PyTorch. This project performs image-to-image translation between two unpaired domains without requiring matching input-output training pairs.

Developed as part of the M.Sc. Cyber Security curriculum (AI / Deep Learning Coursework).

---

## 📌 Features & Architecture

* **Unpaired Image Translation:** Uses cycle-consistency loss ($L_{cyc}$) to learn mappings between Domain A and Domain B.
* **Dual Generators & Discriminators:**
  * $G: A \rightarrow B$ and $F: B \rightarrow A$
  * $D_A$ (discriminates real $A$ vs generated $A$) and $D_B$ (discriminates real $B$ vs generated $B$)
* **Loss Functions:** Adversarial Loss + Cycle-Consistency Loss + Identity Loss.
* **PyTorch Pipeline:** Includes custom dataset loading, modular model architecture, training loops, and inference logic.

---

## 📂 Project Structure

```text
├── models/           # Generator and Discriminator neural network architectures
├── samples/          # Generated output images showing domain translation
├── CycleGan.ipynb    # Interactive Jupyter Notebook implementation
├── CycleGan.py       # Standalone Python training and inference script
├── requirements.txt  # Project dependencies
└── README.md         # Project documentation