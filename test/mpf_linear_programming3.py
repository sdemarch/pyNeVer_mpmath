import sys
import tensorflow as tf
import numpy as np
import random as rn
from mpmath import *

def index_of_max(a):
  l = a[0]
  k = 0
  for i in range(10):
    if (a[i]>l):
      l = a[i]
      k = i
  return k

def get_NN_data():
    my_file1 = open("raw_model/x_value.txt","r")
    x_value = my_file1.readline()
    my_file1.close()
    
    NN_data2 = []
    if (x_value=="1"):
        with open("raw_model/NN_data.txt","r") as e:
            e.readline()
            for line in e:
                if (line!="]\n"):
                    NN_data2.append(mpf(line.replace(',',"")))
    else:
        with open("raw_model/betterNN/best.txt","r") as e:
            e.readline()
            for line in e:
                if (line!="]\n"):
                    NN_data2.append(mpf(line.replace(',',"")))
    return NN_data2
    
def get_label(id):
    g = open("dataset/true_labels2.txt")
    count = 0
    for line in g:
        if (int(count) == int(id)):
            return int(line)
        count += 1

def get_img_final():
    img_final = matrix(28,28)
    T = open("data/x_seed_data.txt","r")
    
    with open("final_safe_value.txt","r") as f:
        for line in f:
            scale_factor = mpf(line)
        
    for i in range(28):
        for j in range(28):
            img_final[i,j] = scale_factor*mpf(T.readline())
                
    T.close()
    return img_final

def main(args):
    mp.dps = args[2]
    
    with open("final_safe_value.txt","r") as f:
        for line in f:
            scale_factor = mpf(line)
        
    if (scale_factor>1.0):
        stat = open("evasion_worked/results"+str(args[1])+".txt","w")
        stat.close()
        return
        
    the_label = get_label(args[1])
    adversarial_label = 0
    if (the_label == 0):
        adversarial_label = 1
    file = open("raw_model/epsilon_value.txt","r")
    base = mpf(file.readline())
    epsilon = mpf(base/255.0)
    file.close()
  
    

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((28,28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='relu'),
        ])
        
    weight_matrix_1 = np.zeros((10, 784))
    bias_1 = np.zeros(10)
    k = 0
    l = 0
    count = 0
    NN_data2 = get_NN_data()
    
    #with open("data/NN_data.txt","r") as Neural:
    for line in range(len(NN_data2)):
        count+=1
        if (k==784 and l<10):
            bias_1[l] = mpf(NN_data2[line])
            l += 1
        if (k<784):
            weight_matrix_1[l][k] = mpf(NN_data2[line])
            l += 1
            if (l==10):
                l=0
                k += 1
    print()
    print()
    for i in range(10):
        for j in range(784):
            indices = [[j,i]]
            model.weights[0].assign(tf.tensor_scatter_nd_update(model.weights[0],indices,[weight_matrix_1[i][j]]))
            
    for i in range(10):
        indices = [[i]]
        model.weights[1].assign(tf.tensor_scatter_nd_update(model.weights[1],indices,[bias_1[i]]))


    img_final = get_img_final()
    print(the_label)
    print(adversarial_label)
    for i in range(28):
        for j in range(28):
            if (weight_matrix_1[the_label][28*i+j]-weight_matrix_1[adversarial_label][28*i+j]>0):
                img_final[i,j] -= epsilon
                if (img_final[i,j]<0):
                    img_final[i,j] = 0
            else:
                img_final[i,j] += epsilon
                if (img_final[i,j]>1):
                    img_final[i,j] = 1
    
    g = open("adversarial_samples/x_adv"+str(args[1])+".ppm","w")
    g.write("P3\n")
    g.write("28 28\n")
    g.write("255\n")
    for aa in range(28):
        for bb in range(28):
            g.write(str(round(img_final[aa,bb]*255.0))+" "+str(round(img_final[aa,bb]*255.0))+" "+str(round(img_final[aa,bb]*255.0))+"\n")
                        
    g.close()
            
    h = open("adversarial_samples/x_adv"+str(args[1])+".txt","w")
    for aa in range(28):
        for bb in range(28):
            h.write(str(img_final[aa,bb])+"\n")
                        
    h.close()
    
if __name__ == "__main__":
    main(sys.argv)
