# pointer_network
--------------------------------------------------------------------
An implementation of Pointer Network using tensorflow. By using sequence to sequence model, we are able to generate a sequence of output from a sequence of input. But the problem of exisiting sequence to sequence model is that the output space is based on vocabulary size. This is a problem in some scenario, where output should only based from input. for example, when we when to sort nature number, the output should only come from number from input. Another example is that when do slot filling, we only care token from input, we do not want to predict token outside of input.
--------------------------------------------------------------------
In this implementation, we provide training and testing method in the file of pointNet.py. you can run train() to train the model, then run predict to make a prediction based on checkpoint.

--------------------------------------------------------------------
Usage(check last lines under PointerNet.py):
--------------------------------------------------------------------
1.train the model
train()

2.make a prediction based on the learned model.
predict()



Tasks:
--------------------------------------------------------------------
Learn to sorting.
for example, give a list[3,5,2,5,6,1], it will output:[5,2,0,3,1,4],
which represent the index of the number in the list,sort by ascending order



References:
--------------------------------------------------------------------
<a href='https://arxiv.org/abs/1506.03134'>Pointer Networks</a>
