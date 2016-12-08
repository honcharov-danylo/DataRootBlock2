import gradient
import math
import random

#actually, i'm again going to use gradient descent in MLE
#okey,data is numerical vector. Why not?

class Function_for_maximize_likelihood_function_Gaussian(gradient.Function_class):
    def __init__(self,data):
        self.data=data

    def Gaussian(self,param,x):
        u=param[0] #mean value
        d=param[1]  #dispersion
        if(d==0):
            return 10000
        first_part=1/(math.sqrt(math.pi*2*d*d))
        power=-pow(x-u,2)/(2*d*d)
        return first_part*math.exp(power)

    def Gaussian_deriv_by_d(self,param,x_value):
        u=param[0] #mean value
        d=param[1]  #dispersion
        return (x_value*x_value-2*x_value*u+u*u-d*d)/pow(d,3)
    def Gaussian_deriv_by_u(self,param,x_value):
        u = param[0]  # mean value
        d = param[1]  # dispersion
        return (x_value-u)/(d*d)
    #let's assume we use only Gaussian
    def function(self,param):
        log_sum=0   #log likelihood function
        for d in self.data:
                log_sum+=math.log(self.Gaussian(param,d))
        return log_sum

    def derivative_by_part_x(self,x,index_of_x,vector_of_x):
        if(index_of_x==0):
            sum=0;
            for d in self.data:
                sum+=self.Gaussian_deriv_by_u(vector_of_x,d)
            #print(sum)
            return sum
        elif(index_of_x==1):
            sum = 0;
            for d in self.data:
                sum += self.Gaussian_deriv_by_d(vector_of_x, d)
            #print(sum)
            return sum
        else: return None


def mle(data):
    func=Function_for_maximize_likelihood_function_Gaussian(data)
    return gradient.gradient_descent(2,3,50,0.00000001,0.01,func,3000,False)


if(__name__=="__main__"):
    testing_vector=[]
    for i in range(500):
        testing_vector.append(random.gauss(4,30))
    print(testing_vector)
    print(mle(testing_vector))
    u_by_formula=sum(testing_vector) / len(testing_vector)
    disp_vector=[pow(i-u_by_formula,2) for i in testing_vector]
    dispersia_by_formula=math.sqrt(sum(disp_vector)/len(disp_vector))
    print("u=",u_by_formula," dispersia=",dispersia_by_formula)