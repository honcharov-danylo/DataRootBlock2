import random
import math
class Function_class:
    def function(self,param):
        return param
    def derivative(self,param):
        return 0
    def derivative_by_part_x(self,x,index_of_x):
        return 0
    def gradient(self,param):
        grad = []
        for p in param:
            grad.append(self.derivative_by_part_x(p))
        return grad

class Function_hypersphere(Function_class): #ok, the function will be f(x1,x2,x3,x4...)=x1^2+x2^2+x3^3....
    def function(self,param):
        sum=0
        for p in param:
            sum+=p*p
        return sum

    def derivative(self,param):
        sum=0
        for p in param:
            sum+=2*p
        return sum

    def derivative_by_part_x(self,x,index_of_x):
        #actually, we don't care about index in this funcion
        return 2*x

def gradient_descent(dimensions,minX,maxX,limit,learning_rate,f):
    values=[]   #initializing
    for i in range(dimensions):
        values.append(random.randrange(minX,maxX))
    cost=f.function(values)
    iteration_counter=0 #a bit of statistics
    while math.fabs(cost)>limit:       #main cycle
        for i in range(len(values)):
            if(f.derivative_by_part_x(values[i],i)>0):
                delta=-1
            else: delta=1
            values[i]=values[i]+learning_rate*delta
        cost=f.function(values)
        iteration_counter+=1
        print("Current values ",values," cost: ",cost)
    print("Grad. descent ended in a ",iteration_counter," iterations.")
    print("Final values of x",values)
    print("Final value of cost function",f.function(values))
    return values

function=Function_hypersphere()
print(gradient_descent(10,-100,100,0.01,0.1,function))
