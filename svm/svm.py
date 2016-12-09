import gradient.gradient as gradient
import regression.regression as regression
from bokeh.plotting import figure, output_file, save,show

import math

class Function_for_svm(gradient.Function_class):
    def __init__(self,model):
        self.model=model

    def function(self, param):
        first_part=sum(param)
        #second_part=0.5*math.pow(sum([a*x*y for y in self.model.keys() for x in self.model[y] for a in param]),2)
        second_part=0
        for l in param:
            for y in self.model.keys():
                for x_vect in self.model[y]:
                    for x in x_vect:
                        second_part+=l*y*x
        return first_part-second_part

    def derivative(self, param):
        return sum([a*y for y in self.model.keys() for a in param])

    def derivative_by_part_x(self, x, index_of_x, vector_of_x):
        sum=0
        for l in vector_of_x:
            for y in self.model.keys():
                for i in range(len(self.model[y])):
                    if i != index_of_x:
                        for x in self.model[y][i]:
                            sum += l * y * x

                        #return sum([a*current_x*y for y in self.model.keys() for current_x in [x_vect for xvect in filter(self.model[y],lambda l: l!=x)] for a in param])
        return sum

    def get_b(self, koefs):
        #b=1/list(model.keys())[0]-sum([x*w for x in self.model[list(self.model.keys())[0]] for w in koefs])
        b=1/list(model.keys())[0]
        for w in koefs:
            for x_vect in self.model[list(self.model.keys())[0]]:
                for x in x_vect:
                    b-=x*w
        return b


if(__name__=="__main__"):
    model=regression.read_model("../svm/points.csv")
    func=Function_for_svm(model)
    #print(model)
    koefs=gradient.gradient_descent(1,-10,10,0.0001,0.1,func)
    b=func.get_b(koefs)
    line_points = {}
    for i in range(70):
        y = koefs[0] * i+b
        line_points[i] = y

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
            color="red", line_width=2)
    # p.line(list(pointsByGradient.keys()), list(pointsByGradient.values()), legend="Linear regression line by gradient.",
    #        color="violet", line_width=2)
    output_file("svm.html")
    # p.line(list(pointsLogistic.keys()),list(pointsLogistic.values()),legend="Logistic.", color="violet",line_width=2)
    save(p)

