"""
EE219 Winter 2017
Project 2 - Task g to Task j

Zeyu Li
lizeyu_cs@foxmail.com
2017-02-12

This is the Task-h "Logistic Regression Classification" (without regularization) of Project 2
In this task, we fit the "comp" data set and "rec" data set separately, and predict the target separately, too.

"""

import project2_toolkit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, classification_report
import matplotlib.pyplot as plt
import os

comp_cate = project2_toolkit.comp_categories
rec_cate  = project2_toolkit.rec_categories

comp_train, comp_train_target, comp_train_name = project2_toolkit.preprocess(comp_cate, subset="train", k=20)
rec_train,  rec_train_target,  rec_train_name  = project2_toolkit.preprocess(rec_cate,  subset="train", k=20)
comp_test, comp_test_target, comp_test_name = project2_toolkit.preprocess(comp_cate, subset="test", k=20)
rec_test,  rec_test_target,  rec_test_name  = project2_toolkit.preprocess(rec_cate,  subset="test", k=20)

# Set C to be super large to eliminate the regularization
comp_logit = LogisticRegression(C=1e5)
comp_logit.fit(comp_train, comp_train_target)
comp_predicted_class = comp_logit.predict(comp_test)
comp_test_target_bin = label_binarize(comp_test_target, classes=[0,1,2,3])
comp_predicted_prob  = comp_logit.predict_log_proba(comp_test)

comp_tpr = dict()
comp_fpr = dict()
for cls in xrange(0, 4):
    comp_fpr[cls], comp_tpr[cls], _ = roc_curve(y_true  = comp_test_target_bin[:, cls],
                                                y_score = comp_predicted_prob[:, cls])

rec_logit = LogisticRegression(C=1e5)
rec_logit.fit(rec_train, rec_train_target)
rec_predicted_class = rec_logit.predict(rec_test)
rec_test_target_bin = label_binarize(rec_test_target, classes=[0,1,2,3])
rec_predicted_prob  = rec_logit.predict_log_proba(rec_test)

rec_tpr = dict()
rec_fpr = dict()
for cls in xrange(0,4):
    rec_fpr[cls], rec_tpr[cls], _ = roc_curve(y_true  = rec_test_target_bin[:, cls],
                                              y_score = rec_predicted_prob[:, cls])

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
plt.title("ROC Curve of Logistic Regression Classifier on Computer articles (without Regularization)")

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
plt.title("ROC Curve of Logistic Regression Classifier on Recreation articles (without Regularization)")

plt.savefig("./logistic_regression_wo_regularization/Roc_curve_logit.png")

comp_report = classification_report(y_pred= comp_predicted_class, y_true= comp_test_target)
rec_report  = classification_report(y_pred= rec_predicted_class,  y_true= rec_test_target)

comp_file = "./logistic_regression_wo_regularization/comp_report.txt"
rec_file  = "./logistic_regression_wo_regularization/rec_report.txt"

try:
    os.remove(comp_file)
    os.remove(rec_file)
except OSError:
    pass

with open(comp_file, 'a') as output:
    output.write(comp_report)
    output.write("\n")
with open(rec_file, 'a')  as output:
    output.write(rec_report)
    output.write("\n")
