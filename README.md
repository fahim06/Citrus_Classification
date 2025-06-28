# Optimizing Citrus Genus Identification

This repository contains the code and findings from the M.Sc. thesis project, "Optimizing Citrus Genus Identification: A Comparative Analysis of Machine Learning-based Methods".The project focuses on developing a robust model to accurately classify different genera of citrus fruits using deep learning techniques, specifically comparing the performance of MobileNet and Inception V3 architectures.

## 📖 Table of Contents

- [Project Overview](#-project-overview)
- [Motivation](#-motivation)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [How to Use](#-how-to-use)
- [Conclusion](#-conclusion)
- [Future Work](#-future-work)
- [Author & Supervisor](#-author--supervisor)

## 📝 Project Overview

This project aims to create an accurate and efficient system for identifying citrus genera from images. Traditional methods for citrus identification are often slow, labor-intensive, and prone to human error. By leveraging state-of-the-art Convolutional Neural Network (CNN) models like MobileNet and Inception V3, this research provides a scalable and automated solution, which is crucial for improving agricultural productivity and quality control.

## 💡 Motivation

Citrus fruits are a significant part of the global agricultural landscape, valued for their nutritional benefits and economic importance. Accurate classification is essential for market value optimization, quality control, and consumer satisfaction. The inefficiency of traditional, manual classification methods poses a major challenge to the citrus industry. This research is driven by the need for an automated, reliable system that can enhance efficiency, ensure consistency, and support the growing demand for traceable and sustainable agricultural products.

## 🖼️ Dataset

The study utilized a comprehensive dataset sourced from [Kaggle](https://www.kaggle.com/datasets/fahimyusuf/citrus/data), consisting of **22,348 images** of eight different citrus genera.

The eight genera included are:

- Limon Criollo
- Limon Mandarino
- Mandarina Pieldesapo
- Mandarina Israeli
- Naranja Valencia
- Tangelo
- Toronja
  The dataset was carefully curated and split into training, validation, and testing sets for rigorous model evaluation.
  

## ⚙️ Methodology

The project follows a systematic workflow from data collection to model deployment, as shown in the diagram below.

![Working Procedure](https://i.imgur.com/your-image-link.png)
_A placeholder for Figure 3.1 from the report_

1.  **Data Collection**: A dataset of 22,348 images was collected from Kaggle.
2.  **Data Preprocessing**: Images underwent rigorous preprocessing, including resizing, normalization, and data augmentation (rotation, flipping, scaling) to enhance model performance and generalization.
3.  **Feature Extraction**: The pre-trained convolutional layers of the CNN models were used as powerful feature extractors, leveraging knowledge from massive image datasets to identify discriminative features in the citrus images.
4.  **Model Training**: Two deep learning models were fine-tuned for the citrus classification task:
    - **MobileNet**: A lightweight and efficient CNN architecture designed for mobile and embedded vision applications.
    - **Inception V3**: A deep CNN architecture known for high performance, built on the Inception module.
5.  **Model Evaluation**: The models were evaluated using standard metrics: accuracy, precision, recall, and F1-score.

## 📊 Results

The performance of the models was evaluated to identify the most suitable architecture for citrus genus identification. **MobileNet** was found to be the best-performing model.

| Model         |  Accuracy  | Precision  |   Recall   | F1-Score |
| :------------ | :--------: | :--------: | :--------: | :------: |
| **MobileNet** | **99.85%** | **99.66%** | **99.52%** | **0.99** |
| Inception V3  | **98.90%** | **96.88**  | **99.32**  | **0.97** |

_[Source: Experimental results demonstrated that MobileNet outperformed the other models in terms of accuracy is 99.85%, precision is 99.66%, recall is 99.52% and Fl-score is 0.99.]_

## 🚀 How to Use

To replicate this project, you will need Python and the libraries listed in the report.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/citrus-genus-identification.git](https://github.com/your-username/citrus-genus-identification.git)
    cd citrus-genus-identification
    ```
2.  **Set up the environment:**
    Install the required libraries (TensorFlow, Keras, NumPy, Pandas, etc.). Refer to Figure 3.2.4.1 for the full list of imports.

3.  **Prepare the dataset:**

    - Download the dataset from Kaggle.
    - Organize it into `train` and `validation` directories as shown in Figure 3.2.4.2.

4.  **Run the models:**
    - Use the provided scripts to define, train, and evaluate the MobileNet and Inception V3 models.

## ✅ Conclusion

his research successfully demonstrates the effectiveness of using deep learning for citrus genus identification. The MobileNet architecture, in particular, provided a robust and highly accurate model, achieving **99.85% accuracy**. The study highlights the importance of data preprocessing and augmentation in achieving high performance. The findings offer a scalable and efficient solution that can significantly benefit the citrus industry by improving quality control and decision-making processes.

## 🔮 Future Work

Future research can expand on this work in several ways:
**Dataset Expansion**: Enlarge the dataset to include more varieties and conditions.
**Real-time Applications**: Develop the model into a real-time classification tool for mobile devices.
**Disease Detection**: Extend the model's capability to include citrus disease detection.
**Comparative Model Analysis**: Compare the performance with other machine learning models like SVM, Decision Tree, Random Forest, etc.

## 👥 Author & Supervisor

**Author**: Fahim Yusuf
**Supervisor**: Anup Majumder, Assistant Professor, Department of Computer Science & Engineering, Jahangirnagar University.

---_This thesis was submitted in partial fulfillment of the requirements for the degree of M.Sc. in Computer Science and Engineering at Jahangirnagar University, Summer 2024._
