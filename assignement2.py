
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples 
from matplotlib import cm



data=pd.read_csv("wines_properties.csv", sep=',' )
data.dropna(how="all", inplace=True)
X=data.iloc[:, 0:13] # we have dropped the last variable, it's categorical and therefore it has no sense to include it in the pca 
X_s= StandardScaler().fit_transform(X)
mean_vector= np.mean(X_s, axis=0)

N=X_s.shape[0]

covariance_matrix= (X_s - mean_vector).T.dot( (X_s - mean_vector))/ (N-1) 

eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix) #eigen vectors will be a numpy ndimensional array 

#now i'll report some data of eigenvalues 
eigen_vectors_values = [ (np.abs(eigen_values[i]), eigen_vectors[:, i], i) for i in range(len(eigen_values))] #now i have a tuple, i can sort, 

 #ora ho autovalore e autovettore

sorted_stuff=sorted(eigen_vectors_values,reverse=True) #it will sort by the first component, mean the eigen values 
tot_variance=sum(eigen_values)
tot_var_explained=[sorted_stuff[i][0]/tot_variance for i in range(len(eigen_values))]
plt.plot(np.cumsum(tot_var_explained))

##could be a good idea to use 3 or 4 dimension 
dim=3
## Creating the eigenvectors matrix (14 x 3)  
top3_eigenvectors = np.hstack( (eigen_vectors_values[0][1].reshape(13,-1),
                             eigen_vectors_values[1][1].reshape(13,-1),
                             eigen_vectors_values[2][1].reshape(13,-1))) 
#projection of the data
Y = X_s.dot(top3_eigenvectors) #on the only 3 dimensions left


top2_eigenvectors = np.hstack( (eigen_vectors_values[0][1].reshape(13,-1),
                             eigen_vectors_values[1][1].reshape(13,-1)))
Y_2 = X_s.dot(top2_eigenvectors)
# analysing the cartesian components of the two eigenvectors related to the main two eigenvalues we see that:


# After having discovered the overall amount of variability explained by each new random variable (represented by the eigen_vectors) we have decided to perform a PCA using the python tool, that automatically gives us the best rotation according to Verimax procedure


my_pca = PCA(n_components=2)
new_projected_data = my_pca.fit_transform(X_s)
PCs = my_pca.components_


# ******************** draw the circle of correlation **************


pca = my_pca.fit(X_s) #need the standardized matrix"

# Get the PCA components (loadings)
PCs = pca.components_

# Use quiver to generate the basic plot
fig = plt.figure(figsize=(5,5))
plt.quiver(np.zeros(PCs.shape[1]), np.zeros(PCs.shape[1]),
           PCs[0,:], PCs[1,:], 
           angles='xy', scale_units='xy', scale=1)

# Add labels based on feature names (here just numbers)
feature_names = np.arange(PCs.shape[1])
for i,j,z in zip(PCs[1,:]+0.02, PCs[0,:]+0.02, feature_names):
    plt.text(j, i, z, ha='center', va='center')

# Add unit circle
circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
plt.gca().add_artist(circle)

# Ensure correct aspect ratio and axis limits
plt.axis('equal')
plt.xlim([-1.0,1.0])
plt.ylim([-1.0,1.0])

# Label axes
plt.xlabel('PC 0')
plt.ylabel('PC 1')

# Done
plt.show()

# comments about it:

#The plot above shows the relationships between all variables. It can be interpreted as follow:
#as we can state from the circle of correlation, (after using varimax method for best rotation) we can say that PC1 account for the  opposite 
#of components 9, 0, 4,2 (color intensity,alcohol, magnesium, ash) while PC0 is positive for high value of 5,6,8 and negative 3,7 (respectively total phenols,flavanoids,Proanthocyanins)
#(ash_alcanity,Nonflavanoid_Phenols )
#Variables 5, 6 and 11 seem to have a correlation higher than 0.4 (the threshold that we use in order to consider them significative) with the PC 0.
#In addition, ositively correlated variables are grouped together, while negatively correlated variables are positioned on opposite quadrants (almost 180 degrees to each other)


# color intensity is a measure of the visual impact of the wine on the taster, and it' sclosely related to PH and alcoholic quantities
# more magnesium, reduce Ph, low levels of proline means poor quality wine 

#overall PC1 is measure of Quality of the wine, while PC0 is related to healthy measures 


# *********************** cluster analysis ******************

#after having permormed our pca, we are using the dimensionally reduced array.
# we could have used the normal data, substituing instead of df the orginal X,
# but computing K_means for 14 components might not have had the same meaningfull result


import pandas as pd
import numpy as np

np.random.seed(123)
labels=[i for i in range(new_projected_data.shape[0])]


df = pd.DataFrame(new_projected_data)

row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')),
                        columns=labels,
                        index=labels)


row_clusters = linkage(df.values, method='complete', metric='euclidean') #mind: i can either pass the entire dataframe, or the distance VECTOR, but not the distance matrix, it won't work

results=pd.DataFrame(row_clusters,
             columns=['row label 1', 'row label 2',
                      'distance', 'no. of items in clust.'],
             index=['cluster %d' % (i + 1)
                    for i in range(row_clusters.shape[0])]) #it will give us the process of forming cluster

row_dendr = dendrogram(row_clusters, 
                       labels=labels
                       )
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()



# alternatively, we could do this 

plt.figure(figsize=(10,7))
dend= shc.dendrogram(shc.linkage(df, method='complete'))




# let's do the analysis over the two dimensions array
X_2=new_projected_data


plt.scatter(X_2[:, 0], X_2[:, 1], 
            c='white', marker='o', edgecolor='black', s=50)
plt.grid()
plt.tight_layout()
plt.show()

#we can manually write this, or either way use the function later built 

km = KMeans(n_clusters=3, 
            init='random', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)

y_km = km.fit_predict(X_2)
num_clust=3
for i in range(num_clust):
    plt.scatter(X_2[y_km == i, 0],
                X_2[y_km == i, 1],
                s=50, c=cm.jet(float(i)/num_clust),
                marker='s', edgecolor='black',
                label="cluster ["+'i'+"]")

# centroids 
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()
    
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]

silhouette_vals = silhouette_samples(X_2, y_km, metric='euclidean')


y_ax_lower, y_ax_upper = 0, 0
yticks = []

for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
             edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--") 

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()
plt.show()

#from the silhouette graph, defined as b(i)-a(i)/(max(a(i),b(i))) we can say that the obtained partition is actually quiete reliable

# the solhouette value is a measure of how close is an object to its own cluster 
#compared to other cluster. An high value (close to one) means that the objext 
#is well matched to its own cluster and poorly matched to neighboring clusters. 
#If all the objects have an high value, the clustering configuration is appropriate.
# In our case the value for each cluster is greater than 0.6, 
#so we can absolutely say that the partition is made properly.

data=pd.read_csv("wines_properties.csv", sep=',')
n_clust=3
data.dropna(how="all", inplace=True)
datas=[]

for i in range(n_clust):
    datas.append((pd.DataFrame.std(data[y_km==i]).values, pd.DataFrame.mean(data[y_km==i]).values))
# i would like to highlight the most important compomnents of the new PC0 and PC1

take_1=[0,2,4,9]
take_0=[3,5,6,7,8]
data_0=[]
data_1=[]
for i in range(n_clust):
    data_0.append((np.take(pd.DataFrame.std(data[y_km==i]).values, take_0), np.take(pd.DataFrame.mean(data[y_km==i]).values, take_0)))
for i in range(n_clust):
    data_1.append((np.take(pd.DataFrame.std(data[y_km==i]).values, take_1), np.take(pd.DataFrame.mean(data[y_km==i]).values, take_1)))




#km.cluster_centers_
 #       [ 2.28888211, -0.95994724],
  #      [-2.73771147, -1.16476397],
   #     [-0.04083933,  1.74320866]
# comments: clusters are   1- Positive PC0 
#                          2- strongly negative PC0
#                          3- fairly high PC1
# we could compute the overall means of the variables, and compare them with the means in our clusters and their variance

# then see if we can detect strong differences
# the first cluster has low levels (on average) of 1 and 12 component (that don't vary much in the cluster, so well concentrated around that mean)
# same way, the second around the 7th component etc







# write the functions 

#in order to choose the best number of cluster, i want to choose the one, between 2 and 10 that has the best silhoutte, 
    # the number is determined by the purpose of the analysis and can vary a lot.
    # if the aim is targeting, definitely no more than 5, but never more than 10 (ottherwise , why should i cluster?)
    
    
def K_means(X): 
    #mind: the function supposes to receive 
    from matplotlib import pyplot as plt
    from sklearn.decomposition import PCA
    
    
    if type(X)!=pd.core.frame.DataFrame:
        raise Exception("non hai inserito un dataframe")
    
    
    # we have decided to project data before performing the cluster
    
    #i'm supposing to receive data after one's PCA, so non categorical values and no string etc
   
    X.dropna(how="all", inplace=True)

    silhouette_avg=[]
    y_km=[]
    
    
    for i in range(2,10):
         km = KMeans(n_clusters=i, 
                init='k-means++', 
                n_init=10, 
                max_iter=300, 
                random_state=0)
         y_km.append(km.fit_predict(X) )   
         silhouette_vals = silhouette_samples(X, y_km[i-2], metric='euclidean')
        
         silhouette_avg.append(np.mean(silhouette_vals))
        
    mass=silhouette_avg[0]
    index=0

    for i in range(len(silhouette_avg)):
        if(mass<silhouette_avg[i]):
            index=i
            mass=silhouette_avg[i]
            
    
    num_clust=2+index
    #now y_km[index] is the vector of the cluster choosen, based on the previous PCA
    my_pca = PCA(n_components=2) 
    X = my_pca.fit_transform(X)
    
    km = KMeans(n_clusters=num_clust, 
                init='k-means++', 
                n_init=10, 
                max_iter=300, 
                random_state=0)
    
    km.fit_predict(X) 
    
    for i in range(num_clust):
            plt.scatter(X[y_km[index] == i, 0],
                        X[y_km[index] == i, 1],
                        s=50, c=cm.jet(float(i)/num_clust),
                        marker='s', edgecolor='black',
                        label="cluster ["+'i'+"]")

    # centroids 
    plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='centroids')
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    return num_clust

#POINT 3
#Write a function that takes in input the dataset: the function performs the PCA and returns the circle 
# of correlations of each pair of principal components (1 and 2, 1 and 3, 1 and ..., 2 and 1, 2 and 3,
#  ...). Plot all the circles in the same plot and/or in a series of plots 3x3.
    


data=pd.read_csv("wines_properties.csv", sep=',' )

data.dropna(how="all", inplace=True)
X=data.iloc[:, 0:13] #********************************* DISCUTIAMONE MOOOOLTO BENE ************************************


def circle_correlation(X):
    #drop what needed, hoping to receive only not categorical and dummy variables:
    
    from sklearn.preprocessing import StandardScaler
    X= StandardScaler().fit_transform(X)
    my_pca = PCA().fit(X)
    PCs = my_pca.components_
    n_rv=X.shape[1]
    
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.5)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    m=0
    
    for h in range(n_rv):
        for k in range(n_rv):
           
            if k<h:
                #i will fill only the lower diagonal
                m+=1
                
                
                if m==10:
                    plt.show()
                    m=1
                    fig = plt.figure()
                    fig.subplots_adjust(hspace=0.5)
                    fig.set_figheight(15)
                    fig.set_figwidth(15)
                plt.subplot(3,3,m)
                plt.quiver(np.zeros(PCs.shape[1]), np.zeros(PCs.shape[1]),
                           PCs[h,:], PCs[k,:], 
                           angles='xy', scale_units='xy', scale=1)
                
                # Add labels based on feature names (here just numbers)
                feature_names = np.arange(PCs.shape[1])
                for i,j,z in zip(PCs[k,:]+0.02, PCs[h,:]+0.02, feature_names):
                    plt.text(j, i, z, ha='center', va='center')
                
                # Add unit circle
                circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
                plt.gca().add_artist(circle)
                
                # Ensure correct aspect ratio and axis limits
                plt.axis('equal')
                plt.xlim([-1.0,1.0])
                plt.ylim([-1.0,1.0])
                
                # Label axes
                plt.xlabel('PC'+str(h))
                plt.ylabel('PC'+str(k))
                
                # Done
    
    plt.show() #plot the remaining stuff


