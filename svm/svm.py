import gradient.gradient as gradient
import numpy as np
import random
import regression.regression as regression
from bokeh.plotting import figure, output_file, save,show

import math

#simple smo realization
#work horrible, but... work
class my_svm:
    def __init__(self,model,C,kernel):
        #self.model=model
        self.model = {i: np.array(model[i]) for i in model.keys()}
        self.C=C
        self.X_array=[]
        self.Y_array=[]
        self.kernel=kernel
        for k in self.model:
            for x_vect in self.model[k]:
                self.X_array.append(x_vect)
                self.Y_array.append(k)
        self.Y_array=np.array(self.Y_array)
        self.X_array=np.array(self.X_array)
        #print(self.X_array," ",self.Y_array)


    def function(self,X):
        return np.sign(np.dot(self.w.T, X.T) + self.b).astype(int)

    def get_w(self,a, y, X):
        return np.dot(a * y, X)

    def train(self,max_passes=20000):
        n=len(self.Y_array)
        a=np.zeros(n)
        self.b=0; #◦ Initialize αi = 0, ∀i, b = 0.
        passes=0 #initialize passes=0

        while (passes < max_passes):
            #num_changed_alphas = 0.
            for i in range(n):
                self.w = self.get_w(a, self.Y_array, self.X_array)
                self.b = self.get_b(self.X_array, self.Y_array, self.w)

                Ei = self.function(self.X_array[i]) - self.Y_array[i]

                j=random.randint(0,n-1)
                while j==i: j=random.randint(0,len(model)-1)
                Ej = self.function(self.X_array[j]) - self.Y_array[j]

                ai_old=a[i] #save old a
                aj_old=a[j]
                y_j,y_i=self.Y_array[j],self.Y_array[i]
                (L, H) = self.compute_L_H(self.C, aj_old, ai_old, y_j, y_i)
                if(L==H):continue
                nu=self.kernel(self.X_array[i],self.X_array[i])+self.kernel(self.X_array[j],self.X_array[j])-\
                2*self.kernel(self.X_array[j],self.X_array[i])
                if nu==0:continue
                a[j] = aj_old + float(y_j * (Ei - Ej)) / nu
                a[j] = max(a[j], L)
                a[j] = min(a[j], H)

                a[i] = ai_old + y_i * y_j * (aj_old - a[j])

                # print(passes," w=",self.w," b=",self.b)
            passes+=1
        self.b = self.get_b(self.X_array, self.Y_array, self.w)#,a)
        self.w = self.get_w(a, self.Y_array, self.X_array)

        return self.w,self.b

    def compute_L_H(self, C, a_j, a_i, y_j, y_i):
        if (y_i != y_j):
            return (max(0, a_j - a_i), min(C, C - a_i + a_j))
        else:
            return (max(0, a_i + a_j - C), min(C, a_i + a_j))

    def get_b(self, X, y, w):
         b_tmp = y - np.dot(w.T, X.T)
         return np.mean(b_tmp)

    def get_b_final(self,X,y,w,a):
        a_v = np.where(a > 0)[0]
        b=1/y[a_v[0]]- np.dot(w.T, X[a_v[0]].T)
        return b

    def kernel_linear(x1, x2):
        return np.dot(x1, x2.T)

#not mine, actually,this thing work better
class Support_Vector_Machine:

    def fit(self, data):
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {}

        transforms = [[1, 1],
                      [-1, 1],
                      [-1, -1],
                      [1, -1]]

        # finding values to work with for our ranges.
        all_data = []
        all_data.extend([x[i][c] for x in self.data.values() for i in range(len(x)) for c in range(len(x[i]))])
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        # no need to keep this memory.
        all_data = None

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # starts getting very high cost after this.
                      self.max_feature_value * 0.001]

        # extremely expensive
        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b] #length of vector

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step

                norms = sorted([n for n in opt_dict])
                # ||w|| : [w,b]
                opt_choice = opt_dict[norms[0]]
                self.w = opt_choice[0]
                self.b = opt_choice[1]
                latest_optimum = opt_choice[0][0] + step * 2
        return (self.w,self.b)


    def predict(self, features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)

        return classification


def line(x, w, b, v):
        return (-w[0] * x - b + v) / w[1]


if(__name__=="__main__"):
    model = regression.read_model("../svm/points.csv")
    mysvm=my_svm(model,1,my_svm.kernel_linear)
    w,b=mysvm.train()
    print("w=", w, " b=", b)

    # data_dict={i:np.array(model[i]) for i in model.keys()}
    # svm=Support_Vector_Machine()
    # w,b=svm.fit(data_dict)
    # print("w=",w," b=",b)

    line_points ={}
    line_points[-20]=line(-20,w,b,0)
    line_points[20] = line(20, w, b, 0)

    first_svm={}
    first_svm[-20] = line(-20, w, b, -1)
    first_svm[20] =line(20, w, b, -1)


    second_svm={}
    second_svm[-20] = line(-20, w, b, 1)
    second_svm[20] = line(20, w, b, 1)

    p = figure(title="SVM", x_axis_label='x', y_axis_label='y')

    points1x=[x[0] for x in model[-1]]
    points1y=[x[1] for x in model[-1]]
    points2x=[x[0] for x in model[1]]
    points2y=[x[1] for x in model[1]]
    p.circle(points1x,points1y, size=20, legend="Points 1.", color="navy",
             alpha=0.5)
    p.circle(points2x, points2y, size=20, legend="Points 2.", color="orange",
             alpha=0.5)

    p.line(list(line_points.keys()), list(line_points.values()), legend="SVM.",
            color="green", line_width=2)

    p.line(list(first_svm.keys()), list(first_svm.values()), legend="support vector=-1",
           color="blue", line_width=1)
    p.line(list(second_svm.keys()), list(second_svm.values()), legend="support vector=1",
           color="gray", line_width=1)

    output_file("svm.html")
    save(p)