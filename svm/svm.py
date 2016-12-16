#doesn't work for now, need's to be repaired. Or rewrited complitely

import gradient.gradient as gradient
import numpy as np
import regression.regression as regression
from bokeh.plotting import figure, output_file, save,show

import math

# class Function_for_svm(gradient.Function_class):
#     def __init__(self,model):
#         self.model=model
#
#     def function(self, param):
#         first_part=sum(param)
#         #second_part=0.5*math.pow(sum([a*x*y for y in self.model.keys() for x in self.model[y] for a in param]),2)
#         second_part=0
#         for l in param:
#             for y in self.model.keys():
#                 for x_vect in self.model[y]:
#                     for x in x_vect:
#                         second_part+=l*y*x
#         return first_part-second_part*second_part
#
#     def derivative(self, param):
#         return sum([a*y for y in self.model.keys() for a in param])
#
#     def derivative_by_part_x(self, x, index_of_x, vector_of_x):
#         sum=0
#         for l in vector_of_x:
#             for y in self.model.keys():
#                 for i in range(len(self.model[y])):
#                     if i != index_of_x:
#                         for x in self.model[y][i]:
#                             sum += l * y * x
#
#                         #return sum([a*current_x*y for y in self.model.keys() for current_x in [x_vect for xvect in filter(self.model[y],lambda l: l!=x)] for a in param])
#         return sum
#
#     def get_w(self,lambdas):
#         w=[]
#         for l in lambdas:
#             for y in self.model.keys():
#                 for x_vect in self.model[y]:
#                     for x in x_vect:
#                         w.append(x*y*l)
#         return w
#     def get_b(self, lambdas):
#         #b=1/list(model.keys())[0]-sum([x*w for x in self.model[list(self.model.keys())[0]] for w in koefs])
#         koefs=self.get_w(lambdas)
#         b=1/list(self.model.keys())[0]
#         for w in koefs:
#             for x_vect in self.model[list(self.model.keys())[0]]:
#                 for x in x_vect:
#                     b-=x*w
#         return b


class Support_Vector_Machine:
    #def __init__(self, visualization=True):
        # self.colors = {1: 'r', -1: 'b'}
        # if self.visualization:
        #     self.fig = plt.figure()
        #     self.ax = self.fig.add_subplot(1, 1, 1)

    # train
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
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

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





if(__name__=="__main__"):
    model = regression.read_model("../svm/points.csv")
    data_dict={i:np.array(model[i]) for i in model.keys()}
    svm=Support_Vector_Machine()
    opt_dict=svm.fit(data_dict)
    print(opt_dict)


    # func=Function_for_svm(model)
    # print(model)
    # koefs=gradient.gradient_descent(1,-10,10,0.00000000001,0.01,func,2000,False)
    # b=func.get_b(koefs)
    # print(b)
    # w=func.get_w(koefs)
    # print(w)

    # line_points = {}
    # for i in range(70):
    #     y = koefs[0] * i+b
    #     line_points[i] = y
    #
    # p = figure(title="SVM", x_axis_label='x', y_axis_label='y')
    #
    # points1x=[x[0] for x in model[-1]]
    # points1y=[x[1] for x in model[-1]]
    # points2x=[x[0] for x in model[1]]
    # points2y=[x[1] for x in model[1]]
    # p.circle(points1x,points1y, size=20, legend="Points 1.", color="navy",
    #          alpha=0.5)
    # p.circle(points2x, points2y, size=20, legend="Points 2.", color="orange",
    #          alpha=0.5)
    #
    # p.line(list(line_points.keys()), list(line_points.values()), legend="SVM.",
    #         color="red", line_width=2)
    # p.line(list(pointsByGradient.keys()), list(pointsByGradient.values()), legend="Linear regression line by gradient.",
    #        color="violet", line_width=2)
    #output_file("svm.html")
    # p.line(list(pointsLogistic.keys()),list(pointsLogistic.values()),legend="Logistic.", color="violet",line_width=2)
    #save(p)

