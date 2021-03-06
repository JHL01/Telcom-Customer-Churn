# Telcom-Customer-Churn
訓練並視覺化評估 9 種不同的機器學習模型：包含探索性資料分析、不平衡資料處理、建立模型、特徵選擇及模型評估。網頁版 [jupyternotebook](https://nbviewer.jupyter.org/github/JHL01/Telcom-Customer-Churn/blob/master/%E9%9B%BB%E4%BF%A1%E5%AE%A2%E6%88%B6%E6%B5%81%E5%A4%B1%28Telco%20Customer%20Churn%29.ipynb)

>### 運作環境：
- Python version == 3.6.4
  - sklearn == 0.20.2
  - numpy == 1.15.4
  - pandas == 0.22.0
  - seaborn == 0.9.0
  - scipy== 1.1.0

>### 專案介紹：
**目標是預測客戶是否在上個月流失(Churn)及找出流失的因素** 。電信公司、網路提供商，有線電視公司和保險公司通常使用客戶流失分析和客戶流失率作為其KPI，因為保留現有客戶的成本遠低於獲得新的客戶。這些行業的公司經常設有客戶服務部門，試圖挽留客戶，因為與新招募的客戶相比，恢復的長期客戶對公司的價值更高。公司通常會區分自願流失和非自願流失:由於客戶決定轉換到另一家公司而發生自願流失及由於客戶搬家而發生非自願流失。在大多數應用中，非自願流失被排除在分析模型之外，因為它並非由公司控制的。由於預測模型能夠生成潛在流失的客群，因此可以有效地將客戶保留行銷方案集中在最易流失的客群中。
>### 資料介紹：
資料來源為 [kaggle](https://www.kaggle.com/blastchar/telco-customer-churn/home) 的公開資料集，由 7,043 筆資料組成，每筆資料有 21 個特徵。'Churn'欄位是預測目標。

>### 方法概述：
首先使用探索性資料分析了解各個特徵與流失率的關係，再用 Scikit-learn 實現 9 種不同算法。其中，由於資料集存在不平衡目標特徵，因此利用生成特徵(SMOTE)加強模型泛化能力與特徵選擇(RFE)嘗試使用較少特徵預測，最後透過學習曲線、混淆矩陣及 ROC 曲線評估模型。

>### 主要發現：
- 使用 SMOTE 後的樣本於 Logistic Regression 進行預測，結果顯示 AUC 從 70.9% 上升至 75.5% ，明顯增加泛化能力。
- 使用 RFE 於 Logistic Regression 時，結果顯示僅使用前 13 個重要特徵時，預測表現與使用所有特徵時接近，表示其餘 8 個特徵提供資訊有限。
- 在不特別調整參數的情況下， Logistic Regression 與 Random Forest 的預測準確度(AUC 及 F-score)最佳，此外，較複雜的模型都有 overfitting 的問題(LGBM Classifier、XGB Classifier 及 MLPClassifier)，顯示在較小的資料集中不需使用複雜的模型。
- 往來時間越長、兩年合約、使用信用卡自動扣款繳費的客人流失率較低；使用紙本帳單、按月合約、總收取金額高的客人流失率較高。
