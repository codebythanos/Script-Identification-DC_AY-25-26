# Script Classification Model Analysis

---

## PARSeq-style ViT Model (Code 1) Performance

The model based on a PARSeq-style Vision Transformer (ViT) achieved an overall accuracy of **81.9%**, indicating strong performance on the script classification task.

The model shows reduced but still noticeable confusion between visually similar scripts such as:

- Hindi–Marathi  
- Tamil–Telugu  
- Bengali–Assamese  

This suggests that while transformer-based global attention improves representation, distinguishing fine-grained structural variations remains challenging.

Certain classes such as Odia and Gujarati achieve high performance, while Marathi continues to show weaker recall, indicating persistent difficulty in learning subtle script differences.

The use of CLS token-based global representation enables better context capture compared to CNN-based models, improving overall classification robustness.

However, full fine-tuning of the ViT backbone increases the risk of overfitting, especially with limited dataset size.

A moderate gap between training and validation accuracy suggests mild overfitting but better generalization compared to earlier CNN-based approaches.

### Main Failure Reasons

- Confusion between structurally similar scripts  
- Limited fine-grained character discrimination  
- Overfitting due to full transformer fine-tuning  

---

## PARSeq-style ViT Model (Code 2) Performance

The improved PARSeq-style ViT model achieved an overall accuracy of **81.8%**, showing performance comparable to the previous model with similar architectural design.

The model continues to exhibit confusion between similar scripts such as:

- Hindi–Marathi  
- Tamil–Telugu  

Despite similar accuracy, this version explores multiple patch configurations, highlighting the importance of token granularity in transformer models.

Class-wise performance remains uneven, with Odia and Punjabi performing strongly, while Marathi remains the weakest class, indicating persistent ambiguity in similar script structures.

The use of multiple patch sizes reveals a trade-off: smaller patches improve detail capture, while larger patches reduce computational complexity but lose fine-grained information.

Training remains stable due to mixed precision and optimized pipeline, but performance gains are limited without deeper architectural changes.

### Main Failure Reasons

- Persistent confusion between visually similar scripts  
- Trade-off between patch size and feature granularity  
- Uneven class-wise performance (Marathi weakest)  

---

## Multi-Patch ViT Model (Code 3) Performance

The Vision Transformer model evaluated with different patch sizes shows the following performance:

- **patch8: 85.4%**  
- **patch16: 86.7% (best)**  
- **patch32: 82.4%**  

This demonstrates that **patch size plays a critical role** in balancing detail and efficiency.

The model still shows confusion between similar scripts, particularly:

- Hindi–Marathi  
- Bengali–Assamese  

The **patch16 configuration achieves the best performance**, indicating an optimal balance between spatial resolution and contextual representation.

Smaller patches (patch8) improve local feature capture but increase computational complexity, while larger patches (patch32) reduce token resolution, leading to weaker discrimination of fine-grained script patterns.

Class-wise imbalance persists, with Marathi remaining the weakest class despite overall improvements, suggesting inherent difficulty in distinguishing it from visually similar scripts.

Despite strong validation trends, slight generalization gaps may still exist due to model complexity and dataset limitations.

### Main Failure Reasons

- Confusion between visually similar scripts  
- Suboptimal token resolution (especially for large patches)  
- Persistent class-wise imbalance (Marathi weakest)  

## PARSeq-Style ViT (128×32) – Failure Analysis

The PARSeq-inspired ViT model using a rectangular 128×32 input shows lower overall performance, with patch8 achieving **79.64% (best)**, patch16 **72.07%**, and patch32 **59.57%**. The model continues to struggle with visually similar scripts such as Hindi–Marathi and Bengali–Assamese.

Smaller patches (patch8) perform better due to higher token resolution, while larger patches (patch32) suffer from significant information loss. The reduced input height (32 pixels) limits vertical detail, affecting fine-grained character recognition. Class-wise imbalance persists, with Marathi remaining the weakest class.

Overall, the performance drop is mainly due to reduced token resolution, loss of spatial detail, and difficulty in distinguishing similar scripts.

### Main Failure Reasons

- Confusion between visually similar scripts (Hindi–Marathi, Bengali–Assamese)  
- Reduced token resolution (especially patch32)  
- Loss of vertical detail (height = 32)  
- Persistent class imbalance (Marathi weakest)  
---


