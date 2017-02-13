"""
EE219 Winter 2017
Project 2 - Task f to Task j

Zeyu Li
lizeyu_cs@foxmail.com
2017-02-07

This is the Task-f "Soft Margin SVM" of Project 2

"""

import project2_toolkit
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np


# Assign the data categories to work on
comp_categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
rec_categories =  ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

# Set the number of components of the SVD as K
K = 50

# Pre-processing the data
comp_train, comp_target, comp_names = project2_toolkit.preprocess(categories=comp_categories, subset="train", k=K)
rec_train,  rec_target,  rec_names  = project2_toolkit.preprocess(categories=rec_categories,  subset="train", k=K)

# Transform the target of computer related as 0, and recreation related as 1
comp_target = [-1 for x in xrange(0, len(comp_target))]
rec_target  = [1 for x in xrange(0, len(rec_target))]

# Combine the two dataset together
# data_set is all the training data from "comp" group and "rec" group
# target_set is all the target of entries from "comp" and "rec" groups
data_set   = np.concatenate((comp_train,  rec_train),  axis=0)
target_set = np.concatenate((comp_target, rec_target), axis=0)

# Setting the Cross validation
kf = KFold(n_splits=5, shuffle=True)

# Setting the regularization C, which is the hyper-parameter on the regularization term
for order_C in range(5):
    for train, test in kf.split(data_set):
        # Setting the C for regularization
        value_C = 10**order_C
        soft_margin_svm = SVC(C=value_C)

        # Fitting the training part
        soft_margin_svm.fit(data_set[train], target_set[train])

        # Predicting on the test data
        predicted = soft_margin_svm.predict(data_set[test])
        report = metrics.classification_report(target_set[test], predicted)
        print report




