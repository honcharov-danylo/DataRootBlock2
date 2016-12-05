#import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, save,show
import csv
def read_model(path):
    model={}
    with open(path, 'r') as csvfile:
        modelreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in modelreader:
            try:
                model[float(row[0])]=float(row[1])
            except ValueError:
                model[row[0]]=row[1]
    return model

def calculate_coef_b1(model):
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
    sum_of_y = 0
    n=len(model)
    for i in range(n):
        sum_of_y += model[list(model.keys())[i]]
    sum_of_x=0
    for i in range(n):
        sum_of_x += list(model.keys())[i]
    return (sum_of_y-b1*sum_of_x)/n

def linear_regression(model):
    b1=calculate_coef_b1(model)
    b0=calculate_coef_b0(model,b1)
    return (b0,b1)



#linear_regression(0)
model=read_model('data_for_linear_regression.csv')
model.pop("X","")
coefs=linear_regression(model)
print(coefs)
points={}
for i in range(len(model)):
    x=list(model.keys())[i]
    y=coefs[0]+coefs[1]*x
    print("x=",x," y=",y)
    points[x]=y

p = figure(title="Linear regression", x_axis_label='x', y_axis_label='y')
p.circle(list(model.keys()),list(model.values()), size=20,legend="Temp.", color="navy", alpha=0.5)
p.line(list(points.keys()),list(points.values()),legend="Temp.", color="red",line_width=2)


save(p)