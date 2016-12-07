#import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, save,show
from gradient import *
from math import exp
import collections
import csv

def read_model(path):
        model={}
        with open(path, 'r') as csvfile:
            modelreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in modelreader:
                x_list = list()
                try:
                    if float((row[len(row)-1])) in model: x_list=model[float(row[len(row)-1])]
                except ValueError:
                    if (row[len(row) - 1]) in model: x_list = model[row[len(row) - 1]]

                x_row_list=list()
                for current in range(len(row)-1):
                    try:
                        x_row_list.append(float(row[current]))
                    except ValueError:
                        x_row_list.append(row[current])

                x_list.append(x_row_list)
                try:
                    model[float(row[len(row)-1])]=x_list
                except ValueError:
                    model[row[len(row) - 1]] = x_list
        return model

def calculate_koef_b1(model):
    model={y[0][0]:x for x,y in model.items()}
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
    model = {y[0][0]: x for x, y in model.items()}
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
            for x_vect in model[m]:
                sum_of_multipl_b_x=param[0]
                for i in range(len(x_vect)):
                    sum_of_multipl_b_x+=x_vect[i]*param[i+1]
                sum+=pow(float(m)-sum_of_multipl_b_x,2)
        return sum

    def derivative(self,param):
        pass #actually, i don't use this in the algorithm, so i prefer not to implement this function

    def derivative_by_part_x(self,x,index_of_x,vector_of_x):
        sum = 0
        n = len(model)
        for m in model.keys():
            for x_vect in model[m]:
                sum_of_line=0
                sum_of_multipl_b_x = vector_of_x[0]
                for i in range(len(x_vect)):
                    sum_of_multipl_b_x += x_vect[i] * vector_of_x[i + 1]
                sum_of_line += float(m) - sum_of_multipl_b_x

                if (index_of_x != 0):
                    sum_of_line *= x_vect[index_of_x - 1]
                sum+=sum_of_line
        sum*=-2
        return sum

def linear_regression(model):
    b1=calculate_koef_b1(model)
    b0=calculate_coef_b0(model,b1)
    return [b0,b1]


def logistic_regression(model,x_vector,koefs):
    power=koefs[0]
    for i in range(len(x_vector)):
        power+=x_vector[i]*koefs[i+1]
    try:
       exponent=exp(power)
    except OverflowError:
       return 0
    probability=exponent/(1+exponent)
    return probability


#linear_regression(0)

if(__name__=="__main__"):

    #working with linear regression now
    model=read_model('data_for_linear_regression.csv')
    model.pop("Y","") #removing some useless stuff
    koefs=linear_regression(model)

    koefs_func=Function_for_calculation_koefs(model)
    koefs_by_grad=list(gradient_descent(2,-10,10,0.1,0.1,koefs_func))
    print("Koefs by gradient ",koefs_by_grad)
    print("Koefs by formulas ",koefs)
    #okey, again this shit. I probably should have wrote the code in the right way wrom beginning, but who cares?
    model_for_plot= {y[0][0]: x for x, y in model.items()}
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
    output_file("linear_regression.html")
    #p.line(list(pointsLogistic.keys()),list(pointsLogistic.values()),legend="Logistic.", color="violet",line_width=2)
    save(p)
    #ok, we have done all this stuff for linear regressin
    #next, logistic:

    model = read_model('data_for_logistic_regression.csv')
    model.pop("CHD","")
    pointsX=[]
    pointsY=[]

    print(model)
    koefs_func=Function_for_calculation_koefs(model)

    koefs = list(gradient_descent(2, -5, 5, 0.1, 0.1, koefs_func))

    for i in range(70):
             pointsX.append(i)
             pointsY.append(logistic_regression(model,[i],koefs))


    p = figure(title="Logistic regression", x_axis_label='x', y_axis_label='y')
    p.line(pointsX, pointsY, legend="Logistic regression line.",
           color="green", line_width=2)

    output_file("logistic_regression.html")
    save(p)
