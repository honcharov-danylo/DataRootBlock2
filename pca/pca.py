import numpy as np
from bokeh.plotting import figure, output_file, save,show

#numpy for covariance matrix and random values. Ok, and eigenvectors,lol. Actually, it could be done by hands, but i prefer not to make things complicated

class PCA:
    def __init__(self,data):
        self.data=data
        self.all_samples=np.concatenate([data[d] for d in data], axis=1)
        #print(self.all_samples.shape)
        #print(self.all_samples)

    def compute_means(self): #need, if we will work with scatter matrix, Actually, we won't
        means=[]
        for k in self.data.keys():
            means.append(sum(self.data[k])/len(self.data[k]))
        return means

    def compute_cov(self):
        cov_mat=np.cov([self.all_samples[i,:] for i in range(self.all_samples.shape[0])])
        #print("Covariation matrix",cov_mat)
        return cov_mat

    def compute_eigen(self):
       return np.linalg.eig(self.compute_cov())

    def sort_eigen(self):
        eigen_values,eigen_vectors=self.compute_eigen()
        pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]
        pairs.sort(key=lambda x: x[0], reverse=True)
        return pairs

    def reduce_dimensions(self):
        pairs=self.sort_eigen()
        matrix=np.hstack([pairs[i][1].reshape(len(pairs[i][1]),1) for i in range(len(pairs)-1)])
        transformed = matrix.T.dot(self.all_samples)
        return transformed

if(__name__=="__main__"):
    size_of_class1=20
    size_of_class2=20
    mu_vec1 = np.array([0, 0, 0])
    cov_mat1 = np.array([[50, 0, 0], [0, 1, 0], [0, 0, 1]])
    class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, size_of_class1).T
    mu_vec2 = np.array([1, 1, 1])
    cov_mat2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, size_of_class2).T
    data={1:class1_sample,-1:class2_sample}
    pca=PCA(data)
    transformed_matrix=pca.reduce_dimensions()

    #visualizing the transformed data, problem is that Bokeh doesn't work with 3d

    p = figure(title="PCA,transformed", x_axis_label='x', y_axis_label='y')
    p.circle(transformed_matrix[0,0:size_of_class1],transformed_matrix[1,0:size_of_class1], size=20, legend="Points 1.", color="navy",
             alpha=0.5)
    p.circle(transformed_matrix[0, size_of_class1:size_of_class1+size_of_class2], transformed_matrix[1, size_of_class1:size_of_class1+size_of_class2],
             size=20,
             legend="Points 2.", color="red",
             alpha=0.5)

    output_file("pca.html")
    save(p)