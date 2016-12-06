#import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, save,show
from gradient import *
from math import exp
from enum import Enum
import csv

def read_model(path):
        model={}
        with open(path, 'r') as csvfile:
            modelreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in modelreader:
                x_list = list()
                for current in range(len(row)-1):
                    try:
                        x_list.append(float(row[current]))
                    except ValueError:
                        x_list.append(row[current])

                try:
                    model[float(row[len(row)-1])]=x_list
                except ValueError:
                    model[row[len(row) - 1]] = x_list
        return model

def calculate_koef_b1(model):
    model={y[0]:x for x,y in model.items()}
    #dirty hack. Very dirty. Firstly the order
    #of elements was like x - key and y - value. The thing is that I want to work
    #with number of x bigger then 1.(This is really hard to explain, I'm not good with words
    #Actually in calculate_koef_b2 the same shit
    n=len(model)
    sum_of_multiplyes_of_x_and_y=0
    for i in range(n):
        sum_of_multiplyes_of_x_and_y+=model[list(model.keys())[i]]*list(model.keys())[i]
    sum_of_multiplyes_of_x_and_y*=n
    sum_of_x=0
    for i in range(n):
        sum_of_x += list(model.keys())[i]
    sum_of_y=0
    for i in range(n):
        sum_of_y += model[list(model.keys())[i]]
    all_x_mult_all_y=sum_of_x*sum_of_y
    numerator=sum_of_multiplyes_of_x_and_y-all_x_mult_all_y
    sum_of_quad_x=0;
    for i in range(n):
        sum_of_quad_x += (list(model.keys())[i])*list(model.keys())[i]
    sum_of_quad_x*=n
    denominator=sum_of_quad_x-sum_of_x*sum_of_x
    return numerator/denominator

def calculate_coef_b0(model,b1):
    model = {y[0]: x for x, y in model.items()}
    sum_of_y = 0
    n=len(model)
    for i in range(n):
        sum_of_y += model[list(model.keys())[i]]
    sum_of_x=0
    for i in range(n):
        sum_of_x += list(model.keys())[i]
    return (sum_of_y-b1*sum_of_x)/n

#if we have only two parameters b0 and b1 we can calculate it.
#Ok, maybe we can calculate it anyway, using calculus, even if we have more than 2 koef
#but we have gradient descent from the previus task. Let's use it,lol
class Function_for_calculation_koefs(Function_class):
    def __init__(self,model):
        self.model=model
    def function(self,param):
        sum=0
        n=len(model)
        for m in model.keys():
            sum_of_multipl_b_x=param[0]
            for i in range(len(model[m])):
                sum_of_multipl_b_x+=model[m][i]*param[i+1]
            sum+=pow(float(m)-sum_of_multipl_b_x,2)
        return sum

    def derivative(self,param):
        pass #actually, i don't use this in the algorithm, so i prefer not to implement this function

    def derivative_by_part_x(self,x,index_of_x,vector_of_x):
        sum = 0
        n = len(model)
        for m in model.keys():
            sum_of_line=0
            sum_of_multipl_b_x = vector_of_x[0]
            for i in range(len(model[m])):
                sum_of_multipl_b_x += model[m][i] * vector_of_x[i + 1]
            sum_of_line += float(m) - sum_of_multipl_b_x

            if (index_of_x != 0):
                sum_of_line *= model[m][index_of_x - 1]
            sum+=sum_of_line

        sum*=-2
        return sum

def linear_regression(model):
    b1=calculate_koef_b1(model)
    b0=calculate_coef_b0(model,b1)
    return [b0,b1]

def logistic_regression(model,x):
    pass
#linear_regression(0)

if(__name__=="__main__"):
    model=read_model('data_for_linear_regression.csv')
    model.pop("Y","")
    koefs=linear_regression(model)

    koefs_func=Function_for_calculation_koefs(model)
    koefs_by_grad=list(gradient_descent(2,-10,10,0.1,0.1,koefs_func))
    print(koefs_by_grad)
    print(koefs)

    #-2.4699999999998865, 17.12999999999971

    #okey, again this shit. I probably should have wrote the code in the right way wrom beginning, but who cares?
    model_for_plot= {y[0]: x for x, y in model.items()}
    pointsByFormula={}
    pointsByGradient={}
    for i in range(len(model_for_plot)):
        x=list(model_for_plot.keys())[i]
        y=koefs[0]+koefs[1]*x
        pointsByFormula[x]=y
        y=koefs_by_grad[0]+koefs_by_grad[1]*x
        pointsByGradient[x]=y

    p = figure(title="Linear regression", x_axis_label='x', y_axis_label='y')
    p.circle(list(model_for_plot.keys()),list(model_for_plot.values()), size=20,legend="Points.", color="navy", alpha=0.5)
    p.line(list(pointsByFormula.keys()), list(pointsByFormula.values()), legend="Linear regression line by formula.", color="red", line_width=2)
    p.line(list(pointsByGradient.keys()), list(pointsByGradient.values()), legend="Linear regression line by gradient.",
           color="violet", line_width=2)

    #p.line(list(pointsLogistic.keys()),list(pointsLogistic.values()),legend="Logistic.", color="violet",line_width=2)
    save(p)