# NetGuard AI - Smart Intrusion Detection System

**Author:** Jayden Trung Nguyen  
**Language:** Python  
**Tools:** TensorFlow, Wireshark, Linux, Cisco Packet Tracer, SQLite  

## Overview
NetGuard AI is a prototype Intrusion Detection System designed for small business networks with limited resources.  
It employs a Long Short-Term Memory (LSTM) model to detect anomalies in network traffic and provide real-time alerts.

## Features
- Processes simulated network traffic datasets captured with Snort and Wireshark  
- Extracts key network features including packet size, protocol, TCP flags, and port information  
- Trains and evaluates an LSTM model achieving **94.7% accuracy** and **0.92 F1-score** on anomaly detection  
- Provides a simple SQLite-based alert logging system and CLI dashboard for easy monitoring by non-technical users  

## How to Run
1. Install required Python packages:  
```bash
pip install pandas numpy tensorflow scikit-learn
```
2. Run the main script:  
```bash
python netguard_ai.py
```

## Example Output
```
[INFO] Training model...
[RESULT] Accuracy: 94.70%
Classification Report:
              precision    recall  f1-score   support
...
```

## Future Improvements
- Integration with live network traffic capture for real-time detection  
- Web-based dashboard with detailed alert visualization  
- Expanded feature engineering and dataset size for improved robustness  
