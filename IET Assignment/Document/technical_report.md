\## Dataset Credits



The dataset used in this project is the \*\*Indian Historical Crop Yield and Weather Dataset\*\*, created and published by \*\*Zoya77\*\* on Kaggle.



\- \*\*Dataset Creator/Owner:\*\* Zoya77

\- \*\*Source:\*\* Kaggle

\- \*\*Dataset:\*\* zoya77/indian-historical-crop-yield-and-weather-data



Full credit for the original dataset, its collection, and preparation belongs to \*\*Zoya77\*\*. This project uses the dataset for academic and research purposes and does not claim ownership of the original dataset.









\# Large-Scale Instance-Based and Statistical Learning Pipeline for Climate-Resilient Crop Yield Forecasting



\## 1. Introduction



This project develops an end-to-end machine learning pipeline for crop-yield forecasting under variable climate conditions. The objective is to support climate-resilient agricultural planning and contribute to SDG 2 (Zero Hunger) and SDG 13 (Climate Action).



The pipeline uses a real multi-year agro-climatic dataset containing crop yield, weather, soil, nutrient, state, district and year information.



All core machine-learning algorithms were implemented from first principles using NumPy and Pandas without using scikit-learn for the core modelling.



\---



\## 2. Dataset



Dataset used:



Indian Historical Crop Yield and Weather Dataset



File:



`data/raw/Custom\_Crops\_yield\_Historical\_Dataset.csv`



The dataset contains:



\- 50,765 records

\- 1966–2017

\- 20 states

\- 311 districts

\- 4 crop categories



Important variables include:



\- Temperature

\- Humidity

\- Rainfall

\- Wind speed

\- Solar radiation

\- Soil pH

\- Area

\- Nutrient requirements

\- Crop yield



The processed dataset is stored at:



`data/processed/clean\_crop\_yield.csv`



\---



\## 3. Data Preprocessing and Feature Engineering



The preprocessing pipeline performs:



1\. Duplicate removal.

2\. Numeric type conversion.

3\. Missing-weather handling using median imputation.

4\. Data validation.

5\. Feature engineering.



Four derived features were created.



\### 3.1 Temperature-Rainfall Index



Temperature-Rainfall Index:



`TRI = Temperature × Rainfall`



This represents the combined influence of temperature and rainfall.



\### 3.2 NPK Index



`NPK Index = N requirement + P requirement + K requirement`



This provides a combined representation of nutrient requirement.



\### 3.3 Climate Stress Index



`Climate Stress = Temperature × (1 - Humidity / 100)`



This provides a simple indicator of climatic stress.



\### 3.4 Production Estimate



`Production Estimate = Area × Yield`



This estimates total crop production in kilograms.



The final processed dataset contains 24 columns.



\---



\## 4. k-Nearest Neighbour Regression



A k-NN regressor was implemented from first principles.



For a query point x, the Euclidean distance is:



`d(x,xi) = sqrt(sum((xj - xij)^2))`



The k nearest training samples are selected and their target values are averaged.



\### 4.1 Mahalanobis Distance



Mahalanobis distance was also implemented:



`d(x,xi) = sqrt((x-xi)^T S^-1 (x-xi))`



where S is the covariance matrix.



A small diagonal regularisation term was added before calculating the pseudoinverse to improve numerical stability.



\### 4.2 k Selection



A manual validation curve was evaluated for:



`k = 1, 3, 5, 7, 9, 11`



The validation experiment showed that the best k according to MSE was:



`k = 1`



The MAE behaviour differed, with k=5 providing lower MAE than k=1. Therefore, MSE was selected as the primary criterion for k selection.



\### 4.3 Distance Metric Comparison



The implemented models produced:



| Metric | MSE | MAE |

|---|---:|---:|

| Euclidean | 174.18 | 4.56 |

| Mahalanobis | 125.78 | 4.28 |



Mahalanobis distance performed better in this experiment, suggesting that accounting for feature covariance was beneficial for this dataset.



\---



\## 5. Locally Weighted Regression



Locally Weighted Regression was implemented from first principles.



For a query point x, the Gaussian weighting function is:



`w\_i = exp(-||x-x\_i||^2 / (2τ^2))`



where τ controls the size of the local neighbourhood.



A weighted least-squares solution is calculated as:



`θ = (X^T W X)^-1 X^T W y`



The prediction is:



`ŷ = x^T θ`



\### 5.1 Bias-Variance Behaviour



A smaller τ gives stronger emphasis to nearby observations.



This generally produces:



\- Lower bias

\- Higher variance

\- Greater sensitivity to noise



A larger τ gives more uniform weights.



This generally produces:



\- Higher bias

\- Lower variance

\- Smoother predictions



Therefore, τ controls the bias-variance trade-off.



The project evaluated:



`τ = 0.1, 0.5, 1.0, 2.0`



The results were saved in:



`results/tables/lwr\_tau\_comparison.csv`



\---



\## 6. Candidate Elimination / Version Space



The Candidate-Elimination component converts continuous climate variables into categorical values:



\- Low

\- Medium

\- High



The variables used were:



\- Temperature

\- Humidity

\- pH

\- Rainfall



Crop yield was converted into:



\- Low risk

\- Medium risk

\- High risk



The analysis used balanced Low and High risk observations.



The resulting experiment produced:



`Specific Boundary (S): \['?', '?', '?', '?']`



and an empty General Boundary.



\### Interpretation



An empty version space means that no hypothesis in the selected hypothesis space perfectly separates the Low- and High-risk observations using only the four discretized climate variables.



This is an important limitation when Candidate Elimination is applied to noisy real-world agricultural data.



\### Inductive Bias



The representation assumes that:



\- continuous variables can be represented by three discrete levels;

\- hypotheses can be expressed using attribute-value constraints;

\- a consistent hypothesis exists within the selected hypothesis space.



Real agricultural systems are more complex because crop yield is affected by many interacting environmental, biological and management factors.



\---



\## 7. Computational Complexity



\### k-NN



For n training records and d features:



Time complexity for one prediction:



`O(n × d)`



Memory complexity:



`O(n × d)`



The main limitation is that k-NN must calculate distances against many training records for each query.



\### Locally Weighted Regression



For each query, LWR calculates weights for all training records and constructs a weighted regression system.



The approximate computational cost grows with the number of training records and the feature dimension. Matrix operations additionally increase computational cost with the number of features.



Memory usage is approximately:



`O(n × d)`



because training observations must be retained.



\---



\## 8. Scalability Analysis



The scalability experiment evaluates:



\- 1,000 records

\- 10,000 records

\- 50,000 records



Theoretical scaling was extended to:



\- 100,000 records

\- 1,000,000 records



The experimental results are stored in:



`results/tables/scalability\_results.csv`



The theoretical results are stored in:



`results/tables/scalability\_theoretical.csv`



For a fixed number of features, increasing the number of records increases the amount of distance computation approximately linearly.



\### 1,000 to 1,000,000 Records



| Records | Relative Computation |

|---:|---:|

| 1,000 | 1× |

| 10,000 | 10× |

| 100,000 | 100× |

| 1,000,000 | 1000× |



Therefore, a direct brute-force k-NN implementation becomes increasingly expensive at million-record scale.



\---



\## 9. Scalability Optimisation Strategy



A spatial indexing or approximate-nearest-neighbour approach can reduce the number of records examined for each query.



The proposed strategy is:



1\. Normalize the feature space.

2\. Organize observations using a spatial index.

3\. Search only the nearby region rather than comparing with every record.

4\. Return the nearest candidate neighbours.



For high-dimensional datasets, approximate-nearest-neighbour hashing can also be considered.



The goal is to reduce the practical search cost while maintaining acceptable prediction accuracy.



\---



\## 10. Visualisations



The project produces more than five visualisations.



\### Visualisation 1: Crop Yield Distribution



Shows the distribution of observed crop yield.



File:



`results/plots/yield\_distribution.png`



\### Visualisation 2: Rainfall vs Crop Yield



Shows the relationship between rainfall and yield.



File:



`results/plots/rainfall\_vs\_yield.png`



\### Visualisation 3: Temperature vs Crop Yield



Shows the relationship between temperature and yield.



File:



`results/plots/temperature\_vs\_yield.png`



\### Visualisation 4: Yearly Yield Trend



Shows average crop yield over time.



File:



`results/plots/yearly\_yield\_trend.png`



\### Visualisation 5: State Yield Comparison



Shows the top states according to average crop yield.



File:



`results/plots/state\_yield\_comparison.png`



\### Visualisation 6: Feature Correlation Matrix



Shows relationships between climate, soil, nutrient and yield variables.



File:



`results/plots/correlation\_matrix.png`



Additional model-result plots include:



\- k-NN validation curve

\- LWR tau comparison



\---



\## 11. Policy Brief



\### Problem



Climate variability creates uncertainty in agricultural production. Farmers and policymakers need evidence-based information to support crop planning and climate adaptation.



\### Key Findings



The analysis demonstrates that climate and agricultural variables can be used to identify relationships with historical crop yield.



The Mahalanobis k-NN experiment produced lower MSE and MAE than Euclidean distance in the evaluated test sample.



Locally Weighted Regression provides a flexible local modelling approach where the bandwidth parameter controls the bias-variance trade-off.



\### Policy Recommendations



1\. Use historical climate-yield relationships as decision-support information rather than deterministic predictions.

2\. Strengthen district-level climate and crop monitoring.

3\. Promote climate-resilient crop planning in regions exposed to rainfall and temperature variability.

4\. Improve agricultural data collection to support more reliable forecasting.

5\. Combine predictive analytics with local agricultural expertise before making high-impact decisions.



\---



\## 12. Limitations and Uncertainty



Historical data cannot perfectly represent future climate conditions.



Major limitations include:



\- Climate distributions may change over time.

\- Historical relationships may not remain stable.

\- Important variables may be missing.

\- Agricultural practices can change.

\- Crop varieties can change.

\- Extreme climate events may be underrepresented.

\- The Candidate-Elimination hypothesis space is restrictive.

\- k-NN becomes computationally expensive for very large datasets.



Predictions should therefore be interpreted probabilistically and as decision-support information rather than guaranteed future outcomes.



\---



\## 13. Fairness Considerations



Forecasting systems may perform differently across regions because agricultural conditions, data availability and crop practices vary.



A model trained predominantly on well-represented regions may provide less reliable predictions for poorly represented districts.



Fairness considerations include:



\- checking performance across regions;

\- identifying areas with limited observations;

\- avoiding decisions based solely on model predictions;

\- maintaining transparency about uncertainty.



\---



\## 14. SDG Relevance



\### SDG 2 – Zero Hunger



The project supports SDG 2 by investigating methods for improving crop-yield forecasting and supporting more informed agricultural planning.



Better yield information can help stakeholders anticipate production changes and improve resource planning.



\### SDG 13 – Climate Action



The project supports SDG 13 by analysing historical relationships between climate variables and agricultural productivity.



The resulting analysis can contribute to climate adaptation and resilient agricultural decision-making.



\---



\## 15. Reproducibility



The project is organised into modular source files:



\- `src/data\_pipeline.py`

\- `src/knn\_regressor.py`

\- `src/locally\_weighted\_regression.py`

\- `src/candidate\_elimination.py`

\- `src/scalability.py`

\- `src/visualization.py`



Automated tests are stored in:



`tests/`



Results are stored in:



`results/`



The complete pipeline can be reproduced using the commands documented in the project README.



\---



\## 16. Conclusion



This project implements a complete from-scratch instance-based and statistical learning pipeline for climate-resilient crop yield analysis.



The system combines data engineering, k-NN regression, Mahalanobis and Euclidean distance metrics, Locally Weighted Regression, Candidate Elimination, scalability analysis, visualisation and policy interpretation.



The experiments demonstrate both the usefulness and limitations of instance-based and local statistical learning for agricultural data. The findings support the use of data-driven analysis as a component of climate-resilient agricultural decision-making while recognising uncertainty and regional differences.






\## Dataset Credits



The dataset used in this project is the \*\*Indian Historical Crop Yield and Weather Dataset\*\*, created and published by \*\*Zoya77\*\* on Kaggle.



\- \*\*Dataset Creator/Owner:\*\* Zoya77

\- \*\*Source:\*\* Kaggle

\- \*\*Dataset:\*\* zoya77/indian-historical-crop-yield-and-weather-data



Full credit for the original dataset, its collection, and preparation belongs to \*\*Zoya77\*\*. This project uses the dataset for academic and research purposes and does not claim ownership of the original dataset.


\## Dataset Credits



The dataset used in this project is the \*\*Indian Historical Crop Yield and Weather Dataset\*\*, created and published by \*\*Zoya77\*\* on Kaggle.



\- \*\*Dataset Creator/Owner:\*\* Zoya77

\- \*\*Source:\*\* Kaggle

\- \*\*Dataset:\*\* zoya77/indian-historical-crop-yield-and-weather-data



Full credit for the original dataset, its collection, and preparation belongs to \*\*Zoya77\*\*. This project uses the dataset for academic and research purposes and does not claim ownership of the original dataset.



\## Dataset Credits



The dataset used in this project is the \*\*Indian Historical Crop Yield and Weather Dataset\*\*, created and published by \*\*Zoya77\*\* on Kaggle.



\- \*\*Dataset Creator/Owner:\*\* Zoya77

\- \*\*Source:\*\* Kaggle

\- \*\*Dataset:\*\* zoya77/indian-historical-crop-yield-and-weather-data



Full credit for the original dataset, its collection, and preparation belongs to \*\*Zoya77\*\*. This project uses the dataset for academic and research purposes and does not claim ownership of the original dataset.



