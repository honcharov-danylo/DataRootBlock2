import math
import random
from bokeh.plotting import figure, output_file, save,show


def dist(point1,point2):
    sm=0
    for i in range(len(point1)):
        sm+=pow(point1[i]-point2[i],2)
    return math.sqrt(sm)

def cluster_points(x_points,centroids):
    clusters={}
    for x_vect in x_points:
        mn=centroids[0]
        for c in centroids:
            if(dist(x_vect,c)<dist(x_vect,mn)):
                mn=c
        if(tuple(mn) in clusters):clusters[tuple(mn)].append(x_vect)
        else: clusters[tuple(mn)]=[x_vect]
    return clusters

def better_centroids(clusters):
    centroids=[]
    for c in clusters:
        current_centr=[]
        for i in range(len(clusters[c][0])):
            i_vect=[x_v[i] for x_v in clusters[c]]
            current_centr.append((sum(i_vect) / len(i_vect)))
        centroids.append(current_centr)
    return centroids

def has_converged(centroids, old_centroids):
    return set([tuple(a) for a in centroids]) == set([tuple(a) for a in old_centroids])

def find_centers(X, K):
    #centroids = random.sample(X, K)
    #centroids_new = random.sample(X, K)
    centroids=[]
    centroids_new=[]
    for i in range(K):
        centroids.append(X[random.randrange(0,len(X))])
        centroids_new.append(X[random.randrange(0, len(X))])
    while not has_converged(centroids_new, centroids):
        centroids = centroids_new
        clusters = cluster_points(X, centroids_new)
        centroids_new = better_centroids(clusters)
    return(centroids_new, clusters)


if(__name__=="__main__"):
    points=[]
    for i in range(200):
        points.append([random.randrange(-50,50),random.randrange(-50,50)])
    centroids_number=7
    centroids,clusters=find_centers(points,centroids_number)

    p = figure(title="Kmeans", x_axis_label='x', y_axis_label='y')

    for c in clusters:
        x_list=[f[0] for f in clusters[c]]
        y_list=[f[1] for f in clusters[c]]

        x_cluster=c[0]
        y_cluster = c[1]

        startColor = [random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)]
        p.circle(x_list, y_list, size=20, legend="Points.", color=tuple(startColor),
             alpha=0.5)
        p.asterisk(x_cluster,y_cluster,size=25,legend="Centroids",color="green")

    # p.circle(points2x, points2y, size=20, legend="Points 2.", color="orange",
    #          alpha=0.5)
    output_file("kmeans.html")
    save(p)