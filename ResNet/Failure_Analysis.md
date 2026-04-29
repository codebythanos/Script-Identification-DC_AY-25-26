## ResNet50 Model1 Performance (Model1.py)

The model based on **ResNet50** achieved an overall accuracy of **53%**, indicating moderate performance on the script classification task.

The model shows significant confusion between visually similar scripts such as:
- Hindi–Marathi  
- Tamil–Telugu  
- Bengali–Assamese  

This suggests a limited ability to capture fine-grained character differences.

Certain classes like **Odia** and **Gujarati** perform well, while **English**, **Hindi**, and **Marathi** exhibit lower precision and recall, indicating inconsistent feature learning across classes.

The use of **vertical splitting** leads to partial or incomplete text regions, causing loss of contextual information and contributing to misclassification.

Although **transfer learning** improves performance, the pretrained features are not fully adapted to script-specific patterns.

A gap between training accuracy (~63%) and validation accuracy (~53%) indicates **mild overfitting** and limited generalization.

---

### Main Failure Reasons
- Confusion between similar scripts  
- Loss of context due to preprocessing  
- Insufficient fine-grained feature learning  




## ResNet50V2 Model2 Performance (Model2.py)

The model based on **ResNet50V2** achieved a higher accuracy of **74.39%**, showing significant improvement over the previous model.

The model still exhibits confusion between structurally similar scripts such as:
- Hindi–Marathi  
- Tamil–Telugu  

Although these errors are reduced compared to earlier models, they are not fully eliminated.

Some classes like **Odia**, **Punjabi**, and **Gujarati** achieve very high performance, while **Marathi** remains the weakest, indicating persistent difficulty in distinguishing similar script patterns.

Despite strong training performance (~88%), validation performance (~67%) shows a noticeable gap, suggesting **overfitting** during fine-tuning.

Heavy **data augmentation** and **class balancing** improve robustness but may also introduce slight distortions in text features.

---

### Main Failure Reasons
- Confusion between visually similar scripts  
- Overfitting due to extensive fine-tuning  
- Uneven class-wise performance (Marathi weakest)  



## ResNet50 with Large-Scale Augmentation (Model3.py)

The model based on **ResNet50** with large-scale augmented data (up to **50K samples per class**) achieved a strong accuracy of **77.56%**, showing clear improvement due to increased data and extended training.

However, the model still shows significant confusion between similar scripts, particularly:
- Hindi–Marathi  
- Bengali–Assamese  

**Marathi** remains the weakest class (~48% recall), while classes like **Odia** and **Punjabi** achieve very high accuracy (>90%). This indicates inconsistent performance across scripts.

Despite very high validation accuracy (~97%), the test accuracy is much lower (~77%), indicating **overfitting** due to heavy data augmentation and synthetic data expansion.

The model may be learning augmented patterns rather than true script-specific features.

---

### Main Failure Reasons
- Persistent confusion between visually similar scripts  
- Overfitting due to large augmented dataset  
- Imbalance in class-wise performance (Marathi weakest)  


<img width="65" height="56" alt="image" src="https://github.com/user-attachments/assets/289510c4-3ccc-4209-a816-145c07c6f813" />
Gujarati misclassified as Hindi

<img width="104" height="72" alt="image" src="https://github.com/user-attachments/assets/d96e31d8-dbb8-42ee-b616-459e782b9aac" />
Kannada misclassfied as English

