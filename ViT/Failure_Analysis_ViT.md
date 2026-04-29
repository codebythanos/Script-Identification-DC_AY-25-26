## Vision Transformer (ViT) Model Performance 

## vit_Model1.py

The model based on **Vision Transformer (ViT)** achieved a test accuracy of **76.92%**, showing strong performance compared to earlier models.

However, the model still shows confusion between visually similar scripts, especially:
- Hindi–Marathi  
- Bengali–Assamese  
- Tamil–Telugu  

**Marathi** remains the weakest class (~55.65%), indicating difficulty in distinguishing structurally similar scripts.

Although validation accuracy reached ~85.6%, the test accuracy is lower (~76.9%), indicating **overfitting** during fine-tuning.

The pretrained backbone is initially frozen and later fully fine-tuned, but the dataset size (~1800 images per class) is still limited for transformer-based models.

Performance is also uneven across classes, with high accuracy for **Odia** and **Gujarati** (>88%) and lower accuracy for **Marathi**, **Bengali**, and **Tamil**, showing imbalance in feature learning.

---

### Main Failure Reasons
- Confusion between similar scripts  
- Overfitting during fine-tuning  
- Limited dataset size for transformer models  




## Vision Transformer (ViT) – Partial Fine-Tuning (vit.Model2.py)

The model based on **Vision Transformer (ViT)** achieved a test accuracy of **58.51%**, showing significantly lower performance compared to the fully fine-tuned ViT model.

The model exhibits strong confusion between visually similar scripts such as:
- Hindi–Marathi  
- Tamil–Telugu  
- Kannada–Telugu  

**Marathi** (~41.63%) and **Telugu** (~46.23%) are the weakest classes, indicating poor fine-grained feature learning.

Although training accuracy improves steadily, validation accuracy saturates around ~66%, and test accuracy drops further, indicating **underfitting** due to limited fine-tuning.

Only the last few transformer layers are unfrozen, which restricts the model’s ability to adapt to script-specific features.

Additionally, heavy **augmentation** and a relatively small dataset (~1800 samples per class) limit the effectiveness of transformer-based learning.

---

### Main Failure Reasons
- Insufficient fine-tuning of the backbone (underfitting)  
- Confusion between similar scripts  
- Limited dataset size for transformer models  




## Vision Transformer (ViT) – Best Model (vit.Model.py)

The model based on **Vision Transformer (ViT)** achieved a high test accuracy of **82.97%**, making it the best-performing model among all.

Despite strong overall performance, the model still shows confusion between visually similar scripts, particularly:
- Hindi–Marathi  
- Bengali–Assamese  

**Marathi** (~63.81%) remains the weakest class, indicating that even advanced models struggle with fine-grained script differences.

A gap between very high training accuracy (~99%) and lower test accuracy (~83%) indicates **overfitting** during full fine-tuning.

Although **hyperparameter tuning** improves results, the dataset size (~1800 samples per class) still limits generalization.

Performance is also slightly uneven across classes, with very high accuracy for **English**, **Gujarati**, and **Punjabi** (>90%) and comparatively lower accuracy for **Marathi** and **Bengali**.

---

### Main Failure Reasons
- Confusion between visually similar scripts  
- Overfitting due to full fine-tuning  
- Limited dataset size for transformer models  
