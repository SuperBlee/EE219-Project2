import project2_toolkit
import numpy as np
import matplotlib.pyplot as plt

# comp_categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
# rec_categories =  ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
#
# comp_train, comp_target, comp_names = project2_toolkit.preprocess(categories=comp_categories, subset="train")
# rec_train,  rec_target,  rec_names  = project2_toolkit.preprocess(categories=rec_categories,  subset="train")
#
# conc = np.concatenate((comp_train, rec_train), axis=0)
# print conc.shape

# plt.figure()
# x = xrange(1,5,1)
# y = xrange(2,6,1)
# plt.plot(x,y)
# plt.savefig("pic-a.png")
#
# y = reversed(y)
#
# plt.savefig("pic-b.png")

import csv

# a = np.array([1,2,3])
# b = np.array([5,7,9])
# d = np.array([1,2,3])
#
# c = np.concatenate((a,b,d), axis=0)
# c = c.reshape((3,3))
#
# np.savetxt("shit.tsv", c, delimiter="\t")

gt = [1,0]
pred = [0.2, 0.1]
gt = np.asarray(gt)
pred = np.asarray(pred)

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_score=pred, y_true=gt)
print fpr
print tpr