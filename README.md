# dynamic pointer_network and pointer network
--------------------------------------------------------------------
Below contain dynamic pointer network and pointer network

Desc of Pointer Network:
--------------------------------------------------------------------
An implementation of Pointer Network using tensorflow. By using sequence to sequence model, we are able to generate a sequence of output from a sequence of input. But the problem of exisiting sequence to sequence model is that the output space is based on vocabulary size. This is a problem in some scenario, where output should only based from input. for example, when we sort nature number, the output should only come from number from input. Another example is that when do slot filling, we only care token from input, we do not want to predict token outside of input.

Desc of Dynamic pointer network:
--------------------------------------------------------------------
Pointer Network provide a pointer mechanism under sequence to sequence framework, to limit output space only from inputs. how can we retreive different information from inputs. for example, if inputs is:" i want to go to new york at 8pm by plane?". if we want to know the time, the answer should be 8pm; if we want to know transporation, it is plane. we invent dynamic pointer network for you to provide your query information, and thus able to retrieve information dynamicly.

For formal description,mechansim and tasks perfomranced, please check <a href='https://github.com/brightmart/dynamic_pointer_network/blob/master/dynamic_pointer_network.pdf'>Dynamic Pointer Network</a>

check: dynamic_pointer_net.py for more informatin

![alt text](https://github.com/brightmart/pointer_network/blob/master/pointer_network.JPG)
![alt text](https://github.com/brightmart/pointer_network/blob/master/pointer_network2.JPG)
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
