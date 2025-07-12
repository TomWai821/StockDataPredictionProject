# Stock Data Prediction Project
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
### 1. Data Source
The experiments use daily OHLCV (Open, High, Low, Close, Volume) data for the NIFTY 50 index constituents, downloaded from Kaggle:
- Source: “Nifty50 Stock Market Data” by Rohan Rao  
- URL: https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data  
- Files: one CSV per ticker (e.g. `ADANIPORTS.csv`)  
- Period: November 2007 – August 2021  
- Columns Include:
   - Date
   - Price indicators: Open, High, Low, Last, Close, VWAP 
   - Volume indicators: Volume, Turnover, Deliverable Volume, %Deliverable
 
#### Image for data
<img src="Image/DataManagement/DataFiltering.png" style="width:60%;"/><br>
Image 1.1 - Filter out not useful data and set csv file data to dataframe<br>

<img src="Image/DataManagement/DataFrameColumnsAndShape.png" style="width:60%;"/><br>
Image 1.2 - Column in dataframe and the shape of dataframe<br>

<img src="Image/DataManagement/DataDescription.png" style="width:50%;"/><br>
Image 1.3 - Data description for each column<br>

<img src="Image/DataManagement/GroupUpData.png" style="width:50%;"/><br>
Image 1.4 - Group up the data to priceData and volumeData (used for data analyst before data training)

### 2. Data Analyst
Analyse the data range, trends and the relation in each column
- Diagram Used:
   - Box Plot (View the range of data)
   - Time Series Map (View the trends of data)
   - Heatmap and pair plot (View the data relation)

#### Image for Data Analyst

**Box plot (Stock Price Data)**:<br>
<img src="Image/Diagrams/AnalyseData/BoxPlot_Open.png" style="width:60%;"/><br>
Image 2.1 - Box plot for Open column data<br>

<img src="Image/Diagrams/AnalyseData/BoxPlot_High.png" style="width:50%;"/><br>
Image 2.2 - Box plot for High column data<br>

<img src="Image/Diagrams/AnalyseData/BoxPlot_Low.png" style="width:50%;"/><br>
Image 2.3 - Box plot for Low column data<br>

<img src="Image/Diagrams/AnalyseData/BoxPlot_PrevClose.png" style="width:50%;"/><br>
Image 2.4 - Box plot for Prev Close column data<br>

<img src="Image/Diagrams/AnalyseData/BoxPlot_Close.png" style="width:50%;"/><br>
Image 2.5 - Box plot for Close column data<br>

<img src="Image/Diagrams/AnalyseData/BoxPlot_VWAP.png" style="width:50%;"/><br>
Image 2.6 - Box plot for VWAP column data<br>

**Box plot (Stock Volume Data)**<br>
<img src="Image/Diagrams/AnalyseData/BoxPlot_Turnover.png" style="width:50%;"/><br>
Image 2.7 - Box plot for Turnover column data<br>

<img src="Image/Diagrams/AnalyseData/BoxPlot_Volume.png" style="width:50%;"/><br>
Image 2.8 - Box plot for Volume column data<br>

<img src="Image/Diagrams/AnalyseData/BoxPlot_DeliverableVolume.png" style="width:50%;"/><br>
Image 2.9 - Box plot for DeliverableVolume column data<br>

<img src="Image/Diagrams/AnalyseData/BoxPlot_PercentageOfDeliverable.png" style="width:50%;"/><br>
Image 2.10 - Box plot for %Deliverable column data<br>

**Time Series Chart (Stock Price Data)**<br>
<img src="Image/Diagrams/AnalyseData/TimeSeriesChart_Open.png" style="width:75%;"/><br>
Image 2.11 - Time Series Chart for Open column data<br>

<img src="Image/Diagrams/AnalyseData/TimeSeriesChart_High.png" style="width:75%;"/><br>
Image 2.12 - Time Series Chart for High column data<br>

<img src="Image/Diagrams/AnalyseData/TimeSeriesChart_Low.png" style="width:75%;"/><br>
Image 2.13 - Time Series Chart for Low column data<br>

<img src="Image/Diagrams/AnalyseData/TimeSeriesChart_PrevClose.png" style="width:75%;"/><br>
Image 2.14 - Time Series Chart for Prev Close column data<br>

<img src="Image/Diagrams/AnalyseData/TimeSeriesChart_Close.png" style="width:75%;"/><br>
Image 2.15 - Time Series Chart for Close column data<br>

<img src="Image/Diagrams/AnalyseData/TimeSeriesChart_VWAP.png" style="width:75%;"/><br>
Image 2.16 - Time Series Chart for VWAP column data<br>

**Time Series Chart (Stock Volume Data)**<br>
<img src="Image/Diagrams/AnalyseData/TimeSeriesChart_Turnover.png" style="width:75%;"/><br>
Image 2.17 - Time Series Chart for Turnover column data<br>

<img src="Image/Diagrams/AnalyseData/TimeSeriesChart_Volume.png" style="width:75%;"/><br>
Image 2.18 - Time Series Chart for Volume column data<br>

<img src="Image/Diagrams/AnalyseData/TimeSeriesChart_DeliverableVolume.png" style="width:75%;"/><br>
Image 2.19 - Time Series Chart for DeliverableVolume column data<br>

<img src="Image/Diagrams/AnalyseData/TimeSeriesChart_PercentageOfDeliverable.png" style="width:75%;"/><br>
Image 2.20 - Time Series Chart for %Deliverable column data<br>

**Pair Plot**<br>
<img src="Image/Diagrams/AnalyseData/PairPlot_PriceData.png" style="width:75%;"/><br>
Image 2.21 - Pair Plot for Stock price related data<br>

<img src="Image/Diagrams/AnalyseData/PairPlot_VolumeData.png" style="width:75%;"/><br>
Image 2.22 - Pair Plot for Stock volume related data<br>

**HeatMap**<br>
<img src="Image/Diagrams/AnalyseData/HeatMap_StockData.png" style="width:75%;"/><br>
Image 2.23 - HeatMap for the whole data<br>

### 3. Preprocessing
1. Scale each column to [0,1] with MinMaxScaler
2. Window into 60-day input sequences for supervised learning
3. Split 80% train / 20% test<br>

#### Image for Preprocessing
<img src="Image/DataManagement/DataProcessingWithMinMaxScaler.png" style="width:50%;"/><br>
Image 3.1 - Apply MinMaxScaler<br>

<img src="Image/DataManagement/FuntionForFixMinMaxScaler.png" style="width:50%;"/><br>
Image 3.2 - Function to apply MinMaxScaler<br>

<img src="Image/\DataManagement/DataLabeling.png" style="width:50%;"/><br>
Image 3.3 - Data Labeling and split to 80% data for training<br>

<img src="Image/DataManagement/FunctionForApplySequences.png" style="width:50%;"/><br>
Image 3.4 - Function For creating sequences<br>

<img src="Image/DataManagement/SetupForDataPredcition.png" style="width:50%;"/><br>
Image 3.5 - Split the Last 20% of the data for the test 


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
