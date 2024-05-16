# Week 1 Diseases Detection with Computer Vision

Summary: Problem facing in medical imaging
Status: Done

## Types of Medical Discipline

**Medical disciplines** encompass various specialized areas within the field of medicine. Each discipline focuses on specific aspects of patient care, research, and practice.

1. **Allergy and Immunology**:
    - Specialists in allergy and immunology work with both adult and pediatric patients suffering from allergies and diseases of the respiratory tract or immune system. They address common conditions such as asthma, food and drug allergies, immune deficiencies, and lung diseases. [Opportunities exist in research, education, and clinical practice1](https://www.sgu.edu/blog/medical/ultimate-list-of-medical-specialties/).
2. **Anesthesiology**:
    - Anesthesiology is dedicated to pain relief for patients before, during, and after surgery. [Subspecialties within anesthesiology include critical care medicine, hospice and palliative care, pain medicine, pediatric anesthesiology, and sleep medicine1](https://www.sgu.edu/blog/medical/ultimate-list-of-medical-specialties/).
3. **Dermatology**:
    - Dermatologists diagnose and treat disorders of the skin, hair, nails, and adjacent mucous membranes. They handle skin cancer, tumors, inflammatory skin diseases, and infectious conditions. [Subspecialties in dermatology include dermatopathology, pediatric dermatology, and procedural dermatology1](https://www.sgu.edu/blog/medical/ultimate-list-of-medical-specialties/).
4. **Diagnostic Radiology**:
    - Physicians specializing in diagnostic radiology use techniques like X-rays, ultrasounds, and MRIs to diagnose illnesses. [They play a crucial role in visualizing internal structures and identifying diseases1](https://www.sgu.edu/blog/medical/ultimate-list-of-medical-specialties/).
5. **Emergency Medicine**:
    - Emergency medicine physicians manage acute medical conditions, trauma, and emergencies. They work in hospital emergency departments and respond to critical situations.
6. **Obstetrics and Gynecology**:
    - Obstetricians and gynecologists focus on women’s reproductive health, pregnancy, childbirth, and related conditions.
7. **Ophthalmology**:
    - Ophthalmologists specialize in eye health, vision, and eye diseases. They perform eye surgeries and manage conditions like cataracts, glaucoma, and retinal disorders.
8. **Pathology**:
    - Pathologists study diseases by examining tissues, cells, and bodily fluids. They play a crucial role in diagnosing cancer and other illnesses.
9. **Pediatrics**:
    - Pediatricians care for infants, children, and adolescents. They address growth, development, and childhood diseases.
10. **Psychiatry**:
    - Psychiatrists diagnose and treat mental health disorders, including depression, anxiety, and schizophrenia.
11. **Surgery**:
    - Surgeons perform surgical procedures, ranging from minor to complex surgeries.
12. **Urology**:
    - Urologists specialize in the urinary system and male reproductive organs. They manage conditions like kidney stones, urinary tract infections, and prostate issues.

## Medical Image Diagnosis

**Medical imaging** plays a crucial role in diagnosing and managing various health conditions. In the world of medical imaging, there are plenty ways to take the medical image, below are some common types of medical imaging techniques:

1. **X-rays (Plain Radiography)**:
    - X-rays use ionizing radiation to create images of bones, organs, and tissues. They are commonly used for detecting fractures, lung infections, and dental issues.
    - A **plain X-ray** of the wrist and hand, for example, can reveal fractures or dislocations.
2. **Computed Tomography (CT)**:
    - CT scans provide detailed cross-sectional images of the body. They are useful for assessing internal structures, detecting tumors, and evaluating trauma.
    - A **CT scan of the chest** can show the heart, lungs, and blood vessels.
3. **Magnetic Resonance Imaging (MRI)**:
    - MRI uses strong magnetic fields and radio waves to create high-resolution images of soft tissues, organs, and the brain.
    - [It is commonly used for brain and spinal cord imaging, as well as assessing joints, muscles, and tumors1](https://www.mayoclinic.org/tests-procedures/mri/about/pac-20384768).
4. **Ultrasound (Sonography)**:
    - Ultrasound uses sound waves to visualize organs, blood vessels, and developing fetuses.
    - It is safe, noninvasive, and commonly used during pregnancy for monitoring fetal development.
5. **Nuclear Medicine Imaging**:
    - Techniques like **positron emission tomography (PET)** and **single-photon emission computed tomography (SPECT)** use radioactive tracers to visualize metabolic processes and detect diseases like cancer.
6. **Fluoroscopy**:
    - Fluoroscopy provides real-time moving images (like X-ray videos) and is used for procedures such as barium swallow studies and angiography.

## Image Classification and Class Imbalance

In the medical imaging field, it's common to have a large number of images with a high imbalance between positive and negative classes. For instance, out of 100,000 CT images, only 1,000 might show a tumor, creating a ratio of 100:1. This significant imbalance can make model training difficult, as the model may simply categorize all images as negative, which would also result in high accuracy.

Thus, it is essential to use strategies that can effectively handle this class imbalance during the model training process. Techniques such as oversampling the minority class, undersampling the majority class, or using a combination of both can be employed. Additionally, the implementation of cost-sensitive methods that assign higher misclassification costs to the minority class can also help in improving the model performance. It's crucial to remember that the goal is not just to achieve high accuracy, but also to ensure that the model can correctly identify both positive and negative cases.

### Weighted Loss

**Weighted loss functions** are commonly used to address the challenge of imbalanced datasets in classification tasks. When dealing with imbalanced data, where the number of instances in one class is significantly smaller than in another class, standard loss functions may lead to biased models that perform poorly on the minority class. 

1. **What Is a Weighted Loss Function?**
    - A weighted loss function modifies the standard loss function used during model training.
    - The idea is to assign a higher penalty to misclassifications of the minority class, making the model more sensitive to that class.
    - Weights are assigned to each class, with higher weight for the minority class and lower weight for the majority class.
2. **Calculating Weights Manually**
    - To calculate weights, you can use the following formula:
        
        ![Untitled](Week%201%20Diseases%20Detection%20with%20Computer%20Vision%20a4080ab27a4a41bab60733fb99989936/Untitled.png)
        
    - Here:
        - `total_samples` is the total number of samples in the dataset.
        - `num_samples_in_class_i` is the number of samples in class `i`.
        - `num_classes` is the total number of classes (in binary classification, it’s 2).
    - For example, if you have a binary classification problem with 1000 samples:
        - 900 samples belong to class 0 (negative class).
        - 100 samples belong to class 1 (positive class).
        - Calculate weights as:
            
            weight_for_class_0=900×21000=0.5556
            
            weight_for_class_1=100×21000=5.0000
            
    - Adjust the weights based on your specific dataset characteristics.
    
    ### Re-sampling with Undersampling and Oversampling
    
    1. **Undersampling**:
        - **Motivation**: Undersampling is used to address imbalanced datasets where one class is significantly underrepresented compared to another class.
        - **How It Works**:
            - In undersampling, we deliberately reduce the number of instances from the majority class (overrepresented class).
            - By doing so, we balance the class distribution, making the model more sensitive to the minority class.
            - For example, if we have a dataset with 90% negative samples and 10% positive samples, we might randomly select fewer negative samples to match the positive sample count.
        - **Use Cases**:
            - Practical scenarios where undersampling is useful include resource constraints (when dealing with large datasets) and when the majority class overwhelms the model’s ability to learn from the minority class.
    2. **Oversampling**:
        - **Motivation**: Oversampling aims to address class imbalance by increasing the representation of the minority class.
        - **How It Works**:
            - In oversampling, we create additional instances of the minority class (often through duplication or synthetic data generation).
            - The model then trains on a more balanced dataset.
            - Techniques like **Random Oversampling**, **SMOTE (Synthetic Minority Oversampling Technique)**, and **ADASYN (Adaptive Synthetic Sampling)** fall under oversampling.
        - **Use Cases**:
            - Oversampling is commonly employed when the detailed data has yet to be collected (e.g., before conducting surveys or interviews).
            - It helps prevent overfitting and ensures that the model learns from both classes effectively.
    3. **Comparison**:
        - **Undersampling** reduces the majority class, while **oversampling** increases the minority class.
        - **Undersampling** may lead to loss of information from the majority class, while **oversampling** introduces synthetic data.
        - **Choosing Between Them**:
            - Use **undersampling** when you have abundant data and want to avoid overfitting.
            - Use **oversampling** when you need to balance the class distribution or when collecting more data is impractical.

## Multi-Task in Medical Imaging

![Untitled](Week%201%20Diseases%20Detection%20with%20Computer%20Vision%20a4080ab27a4a41bab60733fb99989936/Untitled%201.png)

Multitasking is common in the field of medical imaging, as one image can be used to identify various diseases. For instance, an X-ray image can be used to diagnose mass, pneumonia, and edema, as shown in the figure above. The model architecture needs to be modified to handle multiple tasks. The desired output data, Y, can be represented as an array like [0, 1, 0] to indicate the presence or absence of a certain class. The model output, Y_hat, will produce an array of prediction probabilities such as [0.1, 0.8, 0.5]. The loss is calculated by summing up the loss for each class.

This approach is particularly useful when dealing with complex medical cases where multiple diseases or conditions may be present simultaneously. By analyzing a single medical image for multiple disease indicators, the model can provide a more comprehensive diagnosis. This not only saves time but also improves the efficiency of the diagnostic process.

However, it's important to note that as we increase the number of tasks, the complexity of the model also increases. This requires advanced computational resources and sophisticated model architectures to effectively handle the multitasking.

Moreover, the accuracy and reliability of the model's predictions are crucial. False positives or negatives can have serious implications in a medical context. Therefore, it's important to train the model with a large and diverse dataset to improve its predictive accuracy for each task.

In conclusion, multitasking in medical imaging is a powerful approach to disease diagnosis. It allows for a more efficient use of medical images and has the potential to enhance the accuracy and comprehensiveness of medical diagnoses.

![Untitled](Week%201%20Diseases%20Detection%20with%20Computer%20Vision%20a4080ab27a4a41bab60733fb99989936/Untitled%202.png)

## Transfer Learning

![th.jpg](Week%201%20Diseases%20Detection%20with%20Computer%20Vision%20a4080ab27a4a41bab60733fb99989936/th.jpg)

[Certainly**Transfer learning** is a powerful technique in machine learning that leverages knowledge gained from one task or dataset to improve model performance on another related task or a different dataset1](https://builtin.com/data-science/transfer-learning)[2](https://www.datacamp.com/tutorial/transfer-learning). 

1. **Definition**:
    - Transfer learning involves reusing a pre-trained model (often developed for a specific task) as the starting point for building a model on a different task.
    - Instead of training a model from scratch, we transfer the knowledge acquired during one task to enhance generalization in another task.
2. **How Transfer Learning Works**:
    - Imagine you have a neural network that was trained to recognize objects in images (e.g., identifying backpacks).
    - When faced with a new task (e.g., recognizing sunglasses), you can reuse the weights learned by the original model.
    - By transferring these weights, the model already possesses valuable features (such as edge detection or shape recognition) that can benefit the new task.
    - Essentially, we adapt the knowledge from “task A” to improve performance on “task B.”
3. **Benefits of Transfer Learning**:
    - **Data Efficiency**: Transfer learning allows us to train deep neural networks with comparatively little labeled data.
    - **Time Savings**: Starting with pre-trained weights speeds up the training process.
    - **Performance Improvement**: Transfer learning often leads to better generalization and accuracy.
4. **Applications**:
    - **Computer Vision**: Transfer learning is widely used in computer vision tasks. For example, pre-trained models like **ResNet**, **VGG**, or **MobileNet** can be fine-tuned for specific image recognition tasks.
    - **Natural Language Processing (NLP)**: In NLP, transfer learning is applied to tasks like sentiment analysis, text classification, and language modeling. Models like **BERT** and **GPT** are pre-trained on large text corpora and then fine-tuned for specific downstream tasks.
5. **When to Use Transfer Learning**:
    - **Limited Data**: When you have limited labeled data for your target task.
    - **Related Tasks**: When the source and target tasks share some similarities (e.g., both involve image recognition).
    - **Resource Constraints**: Transfer learning is especially useful when computational power or data availability is restricted.
6. **Approaches to Transfer Learning**:
    - **Fine-Tuning**: Adjusting the pre-trained model’s weights on the target task while keeping the initial layers fixed.
    - **Feature Extraction**: Using the pre-trained model’s learned features as input to a new classifier.
    - **Domain Adaptation**: Adapting the model to a different domain (e.g., transferring from synthetic data to real-world data).

## Data Augmentation

1. **What Is Data Augmentation?**:
    - Data augmentation involves creating new training samples by applying various transformations to existing images.
    - By augmenting the dataset, we increase its diversity and help the model generalize better.
2. **Benefits of Data Augmentation in Medical Imaging**:
    - **Increased Dataset Size**: Augmentation generates additional samples, mitigating the impact of limited data.
    - **Improved Generalization**: Augmented data exposes the model to various variations, making it more robust.
    - **Reduced Overfitting**: Augmentation helps prevent the model from memorizing specific examples.
3. **Common Data Augmentation Techniques for Medical Images**:
    - **Rotation**: Rotate the image by a certain angle (e.g., 90 degrees) to simulate different viewpoints.
    - **Translation**: Shift the image horizontally or vertically to introduce positional variations.
    - **Scaling**: Resize the image to different scales (zoom in or out).
    - **Flipping**: Flip the image horizontally or vertically.
    - **Brightness and Contrast Adjustment**: Modify pixel intensity values.
    - **Noise Injection**: Add random noise (e.g., Gaussian noise) to simulate real-world variations.
    - **Elastic Deformation**: Apply local deformations to mimic tissue distortions.
    - **Cropping and Padding**: Crop or pad the image to different sizes.
    - **Histogram Equalization**: Enhance contrast by adjusting pixel intensity histograms.
4. **Application to Different Imaging Modalities**:
    - Data augmentation techniques vary based on the imaging modality (e.g., MRI, CT, mammography, fundoscopy) and the organ being analyzed (brain, lung, breast, eye).
    - Researchers carefully choose augmentation methods based on the specific image type and the deep network architecture used.
5. **Quantitative Performance Evaluations**:
    - Experiments show that the effectiveness of augmentation techniques depends on the image type.
    - Choosing appropriate augmentation strategies is critical for achieving optimal classification results.

**Selecting appropriate data augmentation techniques** is crucial in medical imaging to enhance model performance without altering the essential characteristics of the data. 

1. **Preserving Data Characteristics**:
    - **Nature of Medical Imaging Data**: Medical images (such as X-rays, MRIs, or CT scans) contain valuable diagnostic information. Augmentation should not distort critical features or introduce artifacts.
    - **Domain Knowledge**: Understand the specific imaging modality, anatomical structures, and relevant clinical context. Certain augmentations may be inappropriate for specific organs or pathologies.
2. **Guidelines for Safe Augmentation**:
    - **Rotation**: These transformations are generally safe and preserve anatomical structures. For example:
        - Rotate by small angles (e.g., ±10 degrees).
    - **Scaling and Translation**: Use conservative scaling factors to avoid distorting proportions.
    - **Brightness and Contrast Adjustment**: Adjust within reasonable limits to maintain visibility of important details.
    - **Noise Injection**: Add noise carefully (e.g., Gaussian noise) to simulate realistic variations.
    - **Cropping and Padding**: Crop to regions of interest, but avoid cropping out critical structures.
    - **Elastic Deformation**: Apply locally to mimic tissue deformations without altering overall shape.
3. **Avoiding Harmful Augmentations**:
    - **Geometric Distortions**: Be cautious with extreme rotations, scaling, or shearing. These can alter spatial relationships and affect diagnosis.
    - **Intensity Changes**: Drastic brightness or contrast adjustments may obscure or exaggerate features.
    - **Artifacts**: Augmentations should not introduce artifacts (e.g., ringing, aliasing) that mimic pathology.
4. **Domain-Specific Augmentations**:
    - **Mammography**: Consider augmentations that simulate breast compression or positioning variations.
    - **Brain MRI**: Preserve anatomical landmarks (e.g., ventricles, sulci) during augmentation.
    - **Retinal Images**: Avoid altering vessel patterns or introducing noise near the optic disc.
5. **Validation and Monitoring**:
    - Evaluate augmented images during model validation. Ensure that augmented samples remain diagnostically accurate.
    - Monitor model performance on augmented data to detect any adverse effects.

## Patient Overlap

We must ensure that during the division of the train, validation, and test datasets, a patient's images don't overlap between the train and test set. This is necessary to avoid the model being biased by unique characteristics of specific patients.

![Week%201%20Diseases%20Detection%20with%20Computer%20Vision%20a4080ab27a4a41bab60733fb99989936/Untitled%203.png](Week%201%20Diseases%20Detection%20with%20Computer%20Vision%20a4080ab27a4a41bab60733fb99989936/Untitled%203.png)

The figure above shows a patient's X-ray taken at different times. This patient has a unique characteristic, a necklace, visible in both images. If these images are split into train and test sets, the model might be biased by identifying the presence of the necklace, rather than the presence of diseases.