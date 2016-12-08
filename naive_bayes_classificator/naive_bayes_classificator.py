import regression.regression as regression
from math import log
from collections import defaultdict

class NaiveBayesClassificator():
    def __init__(self,model):
        self.model=model
        self.classes_freq = defaultdict(lambda: 0)
        self.words = defaultdict(lambda: 0)


    def train(self):
        for m in self.model.keys():
            for x_vect in self.model[m]:
                 self.classes_freq[m]+=1
                 for x in x_vect:
                     self.words[m,x]+=1

        for m,x in self.words:                      #normalization
            self.words[m,x]/=self.classes_freq[m]

        for m in self.classes_freq.keys():
            self.classes_freq[m]/=len(model)

    def classificate(self,x_vect):
        class_max_index=0
        class_max_prob=0
        for c in self.classes_freq:
            sum=self.classes_freq[c]
            for word in x_vect:
                sum*=self.words[c, word]
            if(sum>class_max_prob):
                print("Sum is ",sum)
                class_max_index=c
                class_max_prob=sum
        return class_max_index

if(__name__=="__main__"):
    model=regression.read_model("../naive_bayes_classificator/pima-indians-diabetes.csv")
    nb=NaiveBayesClassificator(model)
    nb.train()
    model_test=regression.read_model("../naive_bayes_classificator/test.csv")
    print("First test from the train case: should have 1")
    print(nb.classificate([9,119,80,35,0,29.0,0.263,29]))
    print("Next tests from the test case")
    right_ans=0
    counter=0
    for i in model_test.keys():
        for x_vect in model_test[i]:
            print("Test combination",x_vect)
            print("should have ",i)
            answer=nb.classificate(x_vect)
            print(answer)
            if(answer==i):right_ans+=1
            counter+=1
    print("Accuracy of alg:",right_ans/counter)
    # print("Second test from the test case: should have ", list(model_test.keys())[1])
    # print(nb.classificate(model_test[(list(model_test.keys())[1])][0]))
