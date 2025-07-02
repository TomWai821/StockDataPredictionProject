# StockDataPredictionProject
A comparative study of CNN, RNN and LSTM architectures for stock‐price forecasting using TensorFlow.

## Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Data](#data)   
- [Dependencies](#dependencies)  
- [License](#license)  

## Project Overview
This project benchmarks three deep‐learning approaches—1D Convolutional Neural Networks (CNNs), vanilla Recurrent Neural Networks (RNNs) and Long Short-Term Memory networks (LSTMs)—on historical stock‐price data. The goal is to identify which model achieves the best trade-off between forecasting accuracy and training/inference efficiency.

## Features
- Data ingestion & preprocessing pipeline  
- Parameterized TensorFlow training scripts for CNN, RNN and LSTM  
- Automated evaluation: MSE, MAE, RMSE, direction accuracy  
- Visualization of loss curves and predicted vs. actual prices  

## Installation
1. Clone this repo
   ```bash
   git clone https://github.com/TomWai821/StockDataPredictionProject.git
   cd StockDataPredictionProject

2. Create a virtual environment and install dependencies
   ```bash
   python3 -m venv venv source venv/bin/activate

   # on Windows:
   venv\Scripts\activate pip install -r requirements.txt

## Data
The experiments use daily OHLCV (Open, High, Low, Close, Volume) data for the NIFTY 50 index constituents, downloaded from Kaggle:
- Source: “Nifty50 Stock Market Data” by Rohan Rao  
- URL: https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data  
- Files: one CSV per ticker (e.g. `ADANIPORTS.csv`)  
- Period: November 2007 – August 2021  
- Columns: Date, Open, High, Low, Last, Close, VWAP, Volume, Turnover  

### Preprocessing
1. Forward-fill any missing days  
2. Scale each column to [0,1] with MinMaxScaler  
3. Window into 60-day input sequences for supervised learning  
4. Split 80% train / 20% test  

## Model Architectures
- **CNN**: 1D conv layers + global pooling  
- **RNN**: SimpleRNN layers  
- **LSTM**: Stacked LSTM cells with dropout  

## Dependencies
- TensorFlow  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- scikit-learn  

## License
MIT License

Copyright (c) 2025 TomWai821

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
