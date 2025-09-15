# Sentiment Classification & Clustering - Uber Customer Reviews

## Project Overview

This project analyses **Uber customer reviews** using **Natural Language
Processing (NLP)** techniques.\
The workflow includes sentiment classification, text preprocessing,
embeddings, clustering, and visualisation.

-   **Dataset**: [Uber Customer Reviews (2024) from Kaggle\](https://www.kaggle.com/datasets/kanchana1990/uber-customer-reviews-dataset-2024)
-   **Goal**: Identify customer sentiments and recurring themes (pain
    points & strengths).\
-   **Techniques**: Logistic Regression, TF-IDF, Sentence Embeddings,
    K-Means, DBSCAN\
-   **Visualization**: WordClouds & Power BI dashboard

------------------------------------------------------------------------

## Steps Performed

1.  **Data Loading & Cleaning**

    -   Original shape: `12,000 reviews`
    -   After cleaning: `8,172 reviews` (duplicates & nulls removed)
    -   Key column: `content` (review text)

2.  **Sentiment Labeling**

    -   Based on `score` field:
        -   Positive (4-5 stars)\
        -   Negative (1-2 stars)\
        -   Neutral (3 stars)

    **Distribution:**

    -   Positive: 5,009 (61%)\
    -   Negative: 2,868 (35%)\
    -   Neutral: 295 (4%)

3.  **Text Preprocessing**

    -   Lowercasing, punctuation removal, stopword removal,
        lemmatization.

4.  **Feature Extraction**

    -   TF-IDF (max 3,000 features) for classification\
    -   Sentence embeddings (384-dim) for clustering

5.  **Sentiment Classification (Balanced Logistic Regression)**

    -   Accuracy: **84%**
    -   Positive: F1 = **0.91**
    -   Negative: F1 = **0.83**
    -   Neutral: F1 = **0.14** (struggles due to class imbalance)

    **Confusion Matrix**

        [[474  61  39]
         [ 30  13  16]
         [ 58  55 889]]

    Strong separation for Positive & Negative\
    Weak on Neutral (needs more data)

6.  **Clustering**
   
    #### Elbow & Silhouette Analysis
    The following figure was used to decide the optimal number of clusters:  
    <img width="1397" height="484" alt="results_Sentiment" src="https://github.com/user-attachments/assets/e884bc34-b3f4-4133-bf3d-700fda9596fa" />
    - **Elbow Method:** Suggests possible inflection at *k = 5–6*  
    - **Silhouette Score:** Peaks at *k = 2*, but higher k uncovers richer themes

    - Using sentence embeddings + K-Means

    ### K=2 (Best silhouette)

    -   Cluster 0 (3,955 reviews): Positive experiences (good service,
        great ride)\
        <img width="790" height="427" alt="image" src="https://github.com/user-attachments/assets/87359f53-a000-431b-af78-ced3b14060a9" />

    -   Cluster 1 (4,217 reviews): Mixed issues (drivers, app, pricing,
        customer service)
        <img width="790" height="427" alt="image" src="https://github.com/user-attachments/assets/50fc8a0f-1d42-463e-a88b-0c6f9db91aff" />


    ### K=6 (More granular)

    -   Cluster 0: Driver quality & friendliness\
      <img width="790" height="427" alt="image" src="https://github.com/user-attachments/assets/ecff245a-6dd6-4697-98be-4ff74eb3123f" />

    -   Cluster 1: Operational issues (cancellations, pricing, support)\
      <img width="790" height="427" alt="image" src="https://github.com/user-attachments/assets/4370b20d-b63f-4dac-a196-e433e3879ca2" />

    -   Cluster 2: High service satisfaction\
      <img width="790" height="427" alt="image" src="https://github.com/user-attachments/assets/8bfea42e-02ca-48b4-8cd1-7c284f2c792f" />

    -   Cluster 3: Convenience & affordability\
      <img width="790" height="427" alt="image" src="https://github.com/user-attachments/assets/8e7eabc2-c51b-4fd5-89d0-4c52321c1b11" />

    -   Cluster 4: App usability issues/complaints \
      <img width="790" height="427" alt="image" src="https://github.com/user-attachments/assets/37424e3d-bc6e-45c1-8e42-0f87d7761741" />

    -   Cluster 5: Loyal, very satisfied customers
      <img width="790" height="427" alt="image" src="https://github.com/user-attachments/assets/0e46fb00-c151-4711-a97e-ebb94f2b4686" />

    ### DBSCAN

    -   Clustered ~7800 reviews into 1 dense group, ~342 reviews marked as **noise** (outliers, very short or unique reviews)
    -   Less useful compared to K-Means

8.  **Visualization**

    -   WordClouds generated for each cluster\
    -   Power BI dashboard planned for sentiment distribution & cluster
        insights

------------------------------------------------------------------------

## Key Insights

-   Customers are mostly positive, with strong praise for drivers,
    service quality, and convenience.
-   Major pain points: app usability, cancellations, and pricing.
-   Neutral reviews are underrepresented and require better handling.
-   K=6 clustering reveals actionable business topics:
    -   Improve app experience (Cluster 4)
    -   Address cancellations & pricing transparency (Cluster 1)
    -   Strengthen positive experiences with drivers & service quality

------------------------------------------------------------------------

## Next Steps

-   Fine-tune embeddings with domain-specific models (Uber/transport datasets).
-   Use advanced classifiers (BERT, RoBERTa) to improve Neutral detection.      
-   Deploy Power BI dashboard with:
    -   Sentiment trend over time\
    -   Cluster distribution\
    -   Keyword tracking

------------------------------------------------------------------------

## Outputs

-   WordClouds per cluster: saved in `/wordclouds/`\
-   Model evaluation metrics (classification & clustering)\
-   Ready to use data for Power BI dashboard

------------------------------------------------------------------------

## 👩🏻‍💻Tech Stack

-   Python (Pandas, Scikit-learn, Matplotlib, WordCloud,
    Sentence-Transformers)\
-   Power BI for visualisation \
-   Dataset: \[[Uber Customer Reviews 2024 - Kaggle\]](https://www.kaggle.com/datasets/kanchana1990/uber-customer-reviews-dataset-2024)
