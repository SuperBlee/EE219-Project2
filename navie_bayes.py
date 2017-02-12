"""
EE219 Winter 2017
Project 2 - Task f to Task j

Zeyu Li
lizeyu_cs@foxmail.com
2017-02-07

This is the Task-g "Multinomial Naive Bayes" of Project 2
In this task, we fit the "comp" data set and "rec" data set separately, and predict the target separately, too.

"""

import project2_toolkit
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, classification_report
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

comp_cate = project2_toolkit.comp_categories
rec_cate  = project2_toolkit.rec_categories

# First, fetch all data set from "comp" and "rec", the "train" and the "test"
comp_train, comp_train_target, comp_train_name = project2_toolkit.preprocess(comp_cate, subset="train", k=20)
rec_train,  rec_train_target,  rec_train_name  = project2_toolkit.preprocess(rec_cate,  subset="train", k=20)
comp_test, comp_test_target, comp_test_name = project2_toolkit.preprocess(comp_cate, subset="test", k=20)
rec_test,  rec_test_target,  rec_test_name  = project2_toolkit.preprocess(rec_cate,  subset="test", k=20)

# Do the job on "Computer" class
# Construct an Gaussian NB object, setting the arguments are default
comp_gnb = GaussianNB()
comp_gnb.fit(comp_train, comp_train_target)
comp_predicted_class = comp_gnb.predict(comp_test)
# Binarize the labels of the class
comp_test_target_bin = label_binarize(comp_test_target, classes=[0,1,2,3])
comp_predicted_prob  = comp_gnb.predict_proba(comp_test)

comp_tpr = dict()
comp_fpr = dict()
for cls in xrange(0,4):
    comp_fpr[cls], comp_tpr[cls], _ = roc_curve(y_true  = comp_test_target_bin[:, cls],
                                                y_score = comp_predicted_prob[:, cls])
    
# Repeat the job on "Recreation" class
rec_gnb = GaussianNB()
rec_gnb.fit(rec_train, rec_train_target)
rec_predicted_class = rec_gnb.predict(rec_test)
# Binarize the labels of the class
rec_test_target_bin = label_binarize(rec_test_target, classes=[0,1,2,3])
rec_predicted_prob = rec_gnb.predict_proba(rec_test)

rec_tpr = dict()
rec_fpr = dict()
for cls in xrange(0,4):
    rec_fpr[cls], rec_tpr[cls], _ = roc_curve(y_true= rec_test_target_bin[:, cls],
                                              y_score=rec_predicted_prob[:, cls])

#Plotting the ROC curve of the "Computer" articles
img = plt.figure(figsize=(17, 8))
img.add_subplot(1,2,1)
comp_line0, = plt.plot(comp_fpr[0], comp_tpr[0], lw=2, color='darkorange', label="comp.graphics")
comp_line1, = plt.plot(comp_fpr[1], comp_tpr[1], lw=2, color='blue',       label="comp.os.ms-windows.misc")
comp_line2, = plt.plot(comp_fpr[2], comp_tpr[2], lw=2, color='red',        label="comp.sys.ibm.pc.hardware")
comp_line3, = plt.plot(comp_fpr[3], comp_tpr[3], lw=2, color='green',      label="comp.sys.mac.hardware")
diagonal1,  = plt.plot([0,1],       [0,1],       lw=1, color='black',      linestyle='--')

plt.legend(handles=[comp_line0, comp_line1, comp_line2, comp_line3], loc=4)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve of Gaussian Naive Bayes Classifier on Computer articles")

img.add_subplot(1,2,2)
rec_line0, = plt.plot(rec_fpr[0], rec_tpr[0], lw=2, color='darkorange', label="rec.autos")
rec_line1, = plt.plot(rec_fpr[1], rec_tpr[1], lw=2, color='blue',       label="rec.motorcycles")
rec_line2, = plt.plot(rec_fpr[2], rec_tpr[2], lw=2, color='red',        label="rec.sport.baseball")
rec_line3, = plt.plot(rec_fpr[3], rec_tpr[3], lw=2, color='green',      label="rec.sport.hockey")
diagonal2, = plt.plot([0,1],      [0,1],      lw=1, color='black',      linestyle='--')
plt.legend(handles=[rec_line0, rec_line1, rec_line2, rec_line3], loc=4)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve of Gaussian Naive Bayes Classifier on Recreation articles")

plt.savefig("Roc_curve.png")

comp_report = classification_report(y_pred= comp_predicted_class, y_true= comp_test_target)
rec_report  = classification_report(y_pred= rec_predicted_class,  y_true= rec_test_target)

print rec_report
print comp_report
