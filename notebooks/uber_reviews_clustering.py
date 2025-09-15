import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud

import re
import os

#Config
RANDOM_STATE = 42
WORDCLOUD_DIR = 'wordclouds'
EXPORT_CSV = 'uber_reviews_processed.csv'

if not os.path.exists(WORDCLOUD_DIR):
    os.makedirs(WORDCLOUD_DIR)

#NLTK Setup
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

#Step 1: Load Data
file_path = "XXXXXXXXXX"  # TODO: replace with your file path
df = pd.read_csv(file_path)
print("Shape:", df.shape)
print("Columns:", df.columns)

#Drop NaNs and duplicates
df.dropna(subset=['content'], inplace=True)
df.drop_duplicates(subset=['content'], inplace=True)
print("Shape after cleaning:", df.shape)

#Step 2: Sentiment Labelling
def score_to_sentiment(r):
    try:
        r = float(r)
    except:
        return "Neutral"
    if r <= 2:
        return "Negative"
    elif r == 3:
        return "Neutral"
    else:
        return "Positive"

df['Sentiment'] = df['score'].apply(score_to_sentiment)
print("Sentiment counts:\n", df['Sentiment'].value_counts())

#Step 3: Text Cleaning
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_and_lemmatize(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r"[^a-z\s']", ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

df['Clean_Review'] = df['content'].apply(clean_and_lemmatize)

#Step 4: Baseline Classification
X = df['Clean_Review']
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

#TF-IDF
tfidf_vect = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X_train_tfidf = tfidf_vect.fit_transform(X_train)
X_test_tfidf = tfidf_vect.transform(X_test)

#Logistic Regression with class balancing
clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
clf.fit(X_train_tfidf, y_train)
y_pred = clf.predict(X_test_tfidf)

print("\n*** Classification Report (Balanced Logistic Regression) ***")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=["Negative","Neutral","Positive"]))

#Step 5: Embeddings for Clustering
print("\nGenerating sentence embeddings...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
X_embeddings = embed_model.encode(df['Clean_Review'].tolist(), show_progress_bar=True)
print("Embeddings shape:", X_embeddings.shape)

#Step 6: Choosing Number of Clusters
sil_scores = []
inertias = []
K = range(2, 11)

for k in K:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_embeddings)
    sil = silhouette_score(X_embeddings, km.labels_)
    sil_scores.append(sil)
    inertias.append(km.inertia_)

#Plot Elbow + Silhouette
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].plot(K, inertias, marker='o')
ax[0].set_title("Elbow Method")
ax[0].set_xlabel("Number of clusters (k)")
ax[0].set_ylabel("Inertia")

ax[1].plot(K, sil_scores, marker='o', color="green")
ax[1].set_title("Silhouette Scores")
ax[1].set_xlabel("Number of clusters (k)")
ax[1].set_ylabel("Silhouette Score")

plt.tight_layout()
plt.show()

#Automatically determine the best k based on the highest silhouette score
best_k = K[np.argmax(sil_scores)]
print(f"Best k based on silhouette score: {best_k}")

#Step 7: Perform and Analyse Clustering
def cluster_summary(df, cluster_col, k):
    print(f"\n--- Analyzing {k} Clusters ---")
    for c in sorted(df[cluster_col].unique()):
        cluster_texts = df.loc[df[cluster_col] == c, "Clean_Review"].tolist()
        if not cluster_texts:
            continue
        print(f"\nCluster {c} ({len(cluster_texts)} reviews)")

        wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(cluster_texts))
        plt.figure(figsize=(10,5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"WordCloud - Cluster {c}")
        out_file = os.path.join(WORDCLOUD_DIR, f"wordcloud_k{k}_cluster_{c}.png")
        plt.savefig(out_file, bbox_inches="tight")
        plt.close()
        print(f"Saved WordCloud → {out_file}")

        vect = TfidfVectorizer(max_features=15, stop_words="english")
        tfidf = vect.fit_transform(cluster_texts)
        scores = zip(vect.get_feature_names_out(), np.asarray(tfidf.sum(axis=0)).ravel())
        top_keywords = sorted(scores, key=lambda x: x[1], reverse=True)[:10]
        print("Top keywords:", [w for w,_ in top_keywords])

#Run K-Means for k=best_k and k=6
for k in [best_k, 6]:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    df[f"Cluster_k{k}"] = kmeans.fit_predict(X_embeddings)
    print(f"\nK-Means (k={k}) cluster counts:\n", df[f"Cluster_k{k}"].value_counts())
    cluster_summary(df, f"Cluster_k{k}", k)

#Run DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=8, metric='cosine')
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_embeddings)
print("\nDBSCAN cluster counts (-1 = noise):\n", df['DBSCAN_Cluster'].value_counts().head(10))

#Step 8: Visualise Clusters
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_embeddings)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df[f'Cluster_k{best_k}'], palette='viridis', s=40, alpha=0.8)
plt.title(f'K-Means Clusters (PCA Projection, k={best_k})')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

#Step 9: Export Final Results
df_out = df[["content","score","Sentiment","Clean_Review","Cluster_k2","Cluster_k6","DBSCAN_Cluster"]]
df_out.to_csv(EXPORT_CSV, index=False)
print(f"\nExported final results → {EXPORT_CSV}")
