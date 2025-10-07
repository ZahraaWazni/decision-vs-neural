# 🧠 AI Project - Comparing Machine Learning Models

## 🎓 Context
This project was carried out as part of the **Bachelor’s in Mathematics & Computer Science** (University of Strasbourg), course **Artificial Intelligence**.  
The goal is to design, train, analyze, and compare two families of machine learning models:  
- **Decision Trees**  
- **Artificial Neural Networks**

---

## 📂 Repository Structure
- `src/` → Python scripts  
- `notebooks/` → clean and organized Jupyter notebooks  
- `report.pdf` → written report (answers to project questions)  
- `synthetique.csv` → provided dataset  
- `predictions.zip` → pre-computed predictions (provided for analysis if needed)  
- `auto_evaluation.pdf` → completed self-evaluation form  

---

## ✅ Learning Objectives
- Prepare and analyze a synthetic dataset  
- Implement decision trees and artificial neural networks  
- Apply techniques such as:  
  - **Quartile-based partitioning** (for decision trees)  
  - **Mini-batch learning** and **early stopping** (for neural networks)  
- Compute and interpret classic evaluation metrics:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
- Compare models and argue which one is better  

---

## 🔎 Project Workflow
### 1. Data Preparation
- Split dataset into **80% training / 20% testing**  
- Encoding and normalization if required  
- Check linear separability of the data  

### 2. Model Implementation
#### 🌳 Decision Trees
- Partitioning with **quartiles**  
- Training trees with depths from 3 to 8  
- Keep the **two best-performing models**

#### 🧠 Neural Networks
- Validation set (15% of training) for **early stopping** (patience=4)  
- Training with mini-batches of size 4  
- Tested architectures:  
  - Activation **tanh**: (10,8,6), (10,8,4), (6,4)  
  - Activation **relu**: (10,8,6), (10,8,4), (6,4)  
- Keep the **two best-performing models per activation function**

### 3. Model Analysis
- Manually compute metrics (without external libraries):  
  - Accur
