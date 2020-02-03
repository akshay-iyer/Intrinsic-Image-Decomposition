import matplotlib.pyplot as plt
import numpy as np

y = np.loadtxt('../saved/decomposer/_log_refl_val.txt')
x = range(len(y))


sorted = np.argsort(y)
for i in range(1,20):
	print("{} : {}".format(sorted[-i], y[sorted[-i]]))


# for i in range(len(y)):
# 	if y[i]>10:
# 		y[i] = 10



# plt.plot(x,y,label = "lights_val_loss")
# plt.xlabel('number of epochs')
# plt.ylabel('truncated lights_val_loss')
# plt.legend()
# plt.show()