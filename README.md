# Sentiment Classification & Clustering - Uber Customer Reviews

## 📌 Project Overview

This project analyzes **Uber customer reviews** using **Natural Language
Processing (NLP)** techniques.\
The workflow includes sentiment classification, text preprocessing,
embeddings, clustering, and visualization.

-   **Dataset**: Uber Customer Reviews (2024) from Kaggle\
-   **Goal**: Identify customer sentiments and recurring themes (pain
    points & strengths).\
-   **Techniques**: Logistic Regression, TF-IDF, Sentence Embeddings,
    K-Means, DBSCAN\
-   **Visualization**: WordClouds & Power BI dashboard

------------------------------------------------------------------------

## ⚙️ Steps Performed

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

    ✅ Strong separation for Positive & Negative\
    ⚠️ Weak on Neutral (needs more data or rebalancing)

6.  **Clustering**

    -   Using sentence embeddings + K-Means

    ### K=2 (Best silhouette)

    -   Cluster 0 (3,955 reviews): Positive experiences (good service,
        great ride)\
    -   Cluster 1 (4,217 reviews): Mixed issues (drivers, app, pricing,
        customer service)

    ### K=6 (More granular)

    -   Cluster 0: Driver quality & friendliness\
    -   Cluster 1: Operational issues (cancellations, pricing, support)\
    -   Cluster 2: High service satisfaction\
    -   Cluster 3: Convenience & affordability\
    -   Cluster 4: App usability issues / complaints\
    -   Cluster 5: Loyal, very satisfied customers

    ### DBSCAN

    -   One large cluster (7,830) + noise (342)\
    -   Less useful compared to K-Means

7.  **Visualization**

    -   WordClouds generated for each cluster\
    -   Power BI dashboard planned for sentiment distribution & cluster
        insights

------------------------------------------------------------------------

## 📊 Key Insights

-   Customers are **mostly positive**, with strong praise for **drivers,
    service quality, and convenience**.\
-   Major **pain points**: app usability, cancellations, and pricing.\
-   Neutral reviews are underrepresented and require better handling.\
-   K=6 clustering reveals **actionable business themes**:
    -   Improve app experience (Cluster 4)\
    -   Address cancellations & pricing transparency (Cluster 1)\
    -   Strengthen positive experiences with drivers & service quality

------------------------------------------------------------------------

## 🚀 Next Steps

-   Improve Neutral sentiment detection with oversampling or transformer
    models.\
-   Map sentiment distribution **inside each cluster** for deeper
    insights.\
-   Deploy Power BI dashboard with:
    -   Sentiment trend over time\
    -   Cluster distribution\
    -   Keyword tracking

------------------------------------------------------------------------

## 📁 Outputs

-   WordClouds per cluster: saved in `/wordclouds/`\
-   Model evaluation metrics (classification & clustering)\
-   Ready-to-use data for Power BI dashboard

------------------------------------------------------------------------

## 👨‍💻 Tech Stack

-   Python (Pandas, Scikit-learn, Matplotlib, WordCloud,
    Sentence-Transformers)\
-   Power BI for visualization\
-   Dataset: \[Uber Customer Reviews 2024 - Kaggle\]
