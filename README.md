# **Federated Learning for Image Classification in Distributed Medical Imaging Systems**

## **Overview**  
This project demonstrates the use of **Federated Learning (FL)** to address privacy challenges in medical imaging. Using the **PathMNIST dataset**, we built a decentralized machine learning framework that enables multiple institutions to collaboratively train AI models without sharing sensitive data. By implementing and comparing strategies like **FedAvg**, **FedProx**, and **FedNova**, we achieved robust and accurate image classification while preserving data privacy.

---

## **Key Features**  

### **1. Federated Learning Implementation**  
- **Techniques Used**:
  - **FedAvg**: Baseline federated averaging strategy.  
  - **FedProx**: Adds a proximal term to stabilize training under non-IID data.  
  - **FedNova**: Normalizes client updates to address heterogeneity in local training steps.  
- **Differential Privacy (DP)**: Integrated noise addition and gradient clipping for secure data sharing.  

### **2. CNN Model for Classification**  
- A lightweight **Convolutional Neural Network (CNN)** designed for federated environments:
  - Two convolutional layers with ReLU activation and max-pooling.  
  - Three fully connected layers for classification.  
- Optimized for the **PathMNIST dataset**, classifying histopathological images into **9 tissue types**.  

### **3. PathMNIST Dataset**  
- Dataset contains **107,000 images** resized to **28x28 pixels** with labels representing tissue types:
  - Classes include Adipose, Lympho, Debris, Cancer-Associated Stroma, and others.  
- Supports **IID** and **Non-IID** partitioning to simulate real-world client data distributions.

### **4. Robust Evaluation Metrics**  
- Testing Accuracy:
  - **FedNova**: **86.2%**
  - **FedProx**: **81.3%**
  - **FedAvg**: **79.89%**
- Minimum Loss:
  - **FedNova**: **0.39**
  - **FedProx**: **0.43**
  - **FedAvg**: **0.47**

---

## **Repository Contents**  

1. **`models/`**: CNN architecture and federated learning strategies.   
2. **`docs/`**: Detailed project report with methodology, results, and future work.

---

## **Getting Started**  

### **Prerequisites**  
- Python 3.8+  
- Libraries: `pytorch`, `numpy`, `matplotlib`, `nibabel`, `pandas`, `flwr`  

### **Setup**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/rakeshchary25119/federated-learning-pathmnist.git
   cd federated-learning-pathmnist
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Run the federated learning simulation:  
   ```bash
   python models/run_FL.py
   ```  

---

## **Usage**  

### **Federated Learning Workflow**  
1. **Simulate Clients**: Each client represents a hospital with private data.  
2. **Train Locally**: Local models are trained on private data using specified FL strategies.  
3. **Aggregate Updates**: The central server aggregates updates to form a global model.  
4. **Evaluate Performance**: Test the global model on a centralized test set after every communication round.

### **Visualization**  
- Training accuracy and loss graphs for FedAvg, FedProx, and FedNova strategies.  
- Comparison of predictions for IID and Non-IID data scenarios.  

---

## **Results**  

- **FedNova** emerged as the best-performing strategy, achieving:
  - **Highest testing accuracy**: **86.2%**  
  - **Lowest loss**: **0.39**  
- **FedProx** and **FedAvg** followed, highlighting the impact of data heterogeneity and strategy selection.  
- Differential Privacy effectively secured model updates without significant accuracy loss.  

---

## **Acknowledgments**  
This project was completed under the guidance of **Professor Jianyi Yang** and with the support of **Farzana Yasmin**. Special thanks to my teammates **Sujan Chithaluri** and **Anirudh Kalva** for their collaborative efforts.

---

## **Future Work**  
- Explore advanced optimization techniques like adaptive learning rates for non-IID scenarios.  
- Incorporate communication-efficient methods such as model compression.  
- Extend the framework to handle segmentation tasks on complex medical datasets.  

