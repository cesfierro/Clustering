import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# This is the data exploration step
df = pd.read_csv("DiabetesData2.csv")
print(df.head())
print(df.info())
# Step 1, create scaler object then fit and transform data to pass into our model
scaler = StandardScaler()
scaler.fit(df)
StandardScaler(copy=True, with_mean=True, with_std=True)
samples_scaled = scaler.transform(df)

# Step 2, create kmeans models with the number of clusters from 1-10 to save the inertia values
inertias = []
rng = range(1,11)
for n in rng:
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(samples_scaled)
    inertias.append(kmeans.inertia_)


# Step 3, plot all of the inertia values on graph to compare
plt.plot(rng, inertias, '-o')
plt.xlabel('Number of Clusters, n')
plt.ylabel('Inertia')
plt.xticks(rng)
plt.show()

# Step 4, we found 6 to be our best value now create our model with 6 clusters
kmeans = KMeans(n_clusters = 6)
kmeans.fit(samples_scaled)
# store the labels it predicts
labels = kmeans.predict(samples_scaled)

print("Labels are: ", labels)
print("Inertia value is: ", kmeans.inertia_)
labels_str = ', '.join(map(str, labels))
print("Labels are:", labels_str)
