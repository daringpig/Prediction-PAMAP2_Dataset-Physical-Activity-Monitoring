from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
import numpy as np


####################################### LOAD FILES ###############################################################################################
#Replace with your path
subject_1=sc.textFile("/FileStore/tables/vtz73g0n1485007619747/subject101.dat").map(lambda x : (x.split(" "))).filter(lambda x: x[1] != '0')
subject_2=sc.textFile("/FileStore/tables/xwltpfo71485009157532/subject102.dat").map(lambda x : (x.split(" "))).filter(lambda x: x[1] != '0')
subject_3=sc.textFile("/FileStore/tables/fb97iysm1485009688052/subject103.dat").map(lambda x : (x.split(" "))).filter(lambda x: x[1] != '0')
subject_4=sc.textFile("/FileStore/tables/34lwlxmb1486305557502/subject104.dat").map(lambda x : (x.split(" "))).filter(lambda x: x[1] != '0')
subject_5=sc.textFile("/FileStore/tables/dnxto8z21486305695226/subject105.dat").map(lambda x : (x.split(" "))).filter(lambda x: x[1] != '0')
subject_6=sc.textFile("/FileStore/tables/zshz6k5q1486305920072/subject106.dat").map(lambda x : (x.split(" "))).filter(lambda x: x[1] != '0')
subject_7=sc.textFile("/FileStore/tables/7d0hzi3w1486306152211/subject107.dat").map(lambda x : (x.split(" "))).filter(lambda x: x[1] != '0')
subject_8=sc.textFile("/FileStore/tables/erlpcq3f1486306349416/subject108.dat").map(lambda x : (x.split(" "))).filter(lambda x: x[1] != '0')
subject_9=sc.textFile("/FileStore/tables/rvvlajf01486306627714/subject109.dat").map(lambda x : (x.split(" "))).filter(lambda x: x[1] != '0')

################################################################# FUNCTIONS ################################################################

def sum_line(line):  
#For eache line 
  for i in range(number_of_feature):
    if(line[i]!='NaN'):
      sum_of_a_feature[i].add(float(line[i]))
      
def avg_column(accumulator,number_of_row):
  #Get the average
  avg_feature=[]
  count=0
  for element in sum_of_a_feature:
    avg=element.value/number_of_row
    value=avg
    avg_feature.append(value)
    count+=1
  return avg_feature

def replace_missing_value(line,avg_feature):
  for i in range(number_of_feature):
    if(line[i]=='NaN'):
      line[i]=float(avg_feature[i])
    else:
      line[i]=float(line[i])
  return line

def parsePoint(values):
    #values = [float(x) for x in line]
    return LabeledPoint(values[0], values[1:])
  
def process_subject(subject,number_of_feature,sum_of_a_feature):
  number_of_row=subject.count()
  #Compute sum for each column
  subject.foreach(sum_line)
  #Compute avg of each column
  avg_feature=avg_column(sum_of_a_feature,number_of_row)
  #Replace missing value
  subject_without_missing_value=subject.map(lambda j: replace_missing_value(j,avg_feature))
  #lambda j: processDataLine(j, arg1, arg2)

  return subject_without_missing_value

def insert_label(x):
  result=x[1].tolist()
  result.insert(0,x[0])
  return result


################################################################# MAIN CODE & PROCESS DATA ###########################################
#Process for each subject
number_of_row=subject_1.count()
number_of_feature=54
sum_of_a_feature = [sc.accumulator(0) for x in range(number_of_feature)]
print(number_of_row)

#Replace missing values and compute average each 5 sec 
#subject1=process_subject(subject_1,number_of_feature,sum_of_a_feature)


subject1=process_subject(subject_1,number_of_feature,sum_of_a_feature).map(lambda x:(((x[1],int(x[0]/5)),(np.asarray(x[2:]),1)))).reduceByKey(lambda a,b: (np.sum([a[0],b[0]], axis=0),a[1]+b[1])).map(lambda x: ((x[0][0]),np.true_divide(x[1][0],x[1][1]))).map(insert_label)
subject2=process_subject(subject_2,number_of_feature,sum_of_a_feature).map(lambda x:(((x[1],int(x[0]/5)),(np.asarray(x[2:]),1)))).reduceByKey(lambda a,b: (np.sum([a[0],b[0]], axis=0),a[1]+b[1])).map(lambda x: ((x[0][0]),np.true_divide(x[1][0],x[1][1]))).map(insert_label)
subject3=process_subject(subject_3,number_of_feature,sum_of_a_feature).map(lambda x:(((x[1],int(x[0]/5)),(np.asarray(x[2:]),1)))).reduceByKey(lambda a,b: (np.sum([a[0],b[0]], axis=0),a[1]+b[1])).map(lambda x: ((x[0][0]),np.true_divide(x[1][0],x[1][1]))).map(insert_label)
subject4=process_subject(subject_4,number_of_feature,sum_of_a_feature).map(lambda x:(((x[1],int(x[0]/5)),(np.asarray(x[2:]),1)))).reduceByKey(lambda a,b: (np.sum([a[0],b[0]], axis=0),a[1]+b[1])).map(lambda x: ((x[0][0]),np.true_divide(x[1][0],x[1][1]))).map(insert_label)
subject5=process_subject(subject_5,number_of_feature,sum_of_a_feature).map(lambda x:(((x[1],int(x[0]/5)),(np.asarray(x[2:]),1)))).reduceByKey(lambda a,b: (np.sum([a[0],b[0]], axis=0),a[1]+b[1])).map(lambda x: ((x[0][0]),np.true_divide(x[1][0],x[1][1]))).map(insert_label)
subject6=process_subject(subject_6,number_of_feature,sum_of_a_feature).map(lambda x:(((x[1],int(x[0]/5)),(np.asarray(x[2:]),1)))).reduceByKey(lambda a,b: (np.sum([a[0],b[0]], axis=0),a[1]+b[1])).map(lambda x: ((x[0][0]),np.true_divide(x[1][0],x[1][1]))).map(insert_label)
subject7=process_subject(subject_7,number_of_feature,sum_of_a_feature).map(lambda x:(((x[1],int(x[0]/5)),(np.asarray(x[2:]),1)))).reduceByKey(lambda a,b: (np.sum([a[0],b[0]], axis=0),a[1]+b[1])).map(lambda x: ((x[0][0]),np.true_divide(x[1][0],x[1][1]))).map(insert_label)
subject8=process_subject(subject_8,number_of_feature,sum_of_a_feature).map(lambda x:(((x[1],int(x[0]/5)),(np.asarray(x[2:]),1)))).reduceByKey(lambda a,b: (np.sum([a[0],b[0]], axis=0),a[1]+b[1])).map(lambda x: ((x[0][0]),np.true_divide(x[1][0],x[1][1]))).map(insert_label)
#subject9=process_subject(subject_9,number_of_feature,sum_of_a_feature).map(lambda x:(((x[1],int(x[0]/5)),(np.asarray(x[2:]),1)))).reduceByKey(lambda a,b: (np.sum([a[0],b[0]], axis=0),a[1]+b[1])).map(lambda x: ((x[0][0]),np.true_divide(x[1][0],x[1][1]))).map(insert_label)



#subject_rdd_list=[subject1,subject2,subject3,subject4,subject5,subject6,subject7,subject8,subject9]
subject_rdd_list=[subject1,subject2,subject3,subject4,subject5,subject6,subject7,subject8]


list_error_rf=[]
list_error_lr=[]
list_error_dt=[]

#LEAVE ONE OUT 

for i in range(len(subject_rdd_list)):
  #subject_rdd_list=[subject1,subject2,subject3,subject4,subject5,subject6,subject7,subject8,subject9]
  subject_rdd_list=[subject1,subject2,subject3,subject4,subject5,subject6,subject7,subject8]
  test_subject=subject_rdd_list[i]

  del subject_rdd_list[i]

  dataset_subject_without_missing_value=sc.union(subject_rdd_list)
  #Load and parse the data file into an RDD of LabeledPoint.
  data = dataset_subject_without_missing_value.map(parsePoint)
  # Split the data into training and test sets (30% held out for testing)
  trainingData = data #All the subject except 1
  testData = test_subject.map(parsePoint)  #On the subject leaves


################################################################# MODEL #############################################################################
  ###########Train a RandomForest model.######################
  #  Empty categoricalFeaturesInfo indicates all features are continuous.
  #  Note: Use larger numTrees in practice.
  #  Setting featureSubsetStrategy="auto" lets the algorithm choose.
  model = RandomForest.trainClassifier(trainingData, numClasses=25, categoricalFeaturesInfo={},
                                       numTrees=2000, featureSubsetStrategy="auto",
                                       impurity='gini', maxDepth=5, maxBins=32)
  #Evaluate model on test instances and compute test error
  predictions = model.predict(testData.map(lambda x: x.features))
  labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
  testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
  #Save the error on the leave one out 
  list_error_rf.append(testErr) 
  #print('Test Error = ' + str(testErr))


  ########### Logisitic Regression model.######################
  model_regression = LogisticRegressionWithLBFGS.train(trainingData,numClasses=25)
  # Evaluating the model on testing data
  labelsAndPreds = testData.map(lambda p: (p.label, model_regression.predict(p.features)))
  trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(data.count())
  list_error_lr.append(testErr)
  #print("Training Error = " + str(trainErr))


   ########### Decision Tree .######################
  #Empty categoricalFeaturesInfo indicates all features are continuous.
  model = DecisionTree.trainClassifier(trainingData, numClasses=25, categoricalFeaturesInfo={},
                                     impurity='gini', maxDepth=5, maxBins=32)

  # Evaluate model on test instances and compute test error
  predictions = model.predict(testData.map(lambda x: x.features))
  labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
  testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
  list_error_dt.append(testErr)
  #print('Test Error = ' + str(testErr))
    #print(model.toDebugString())
        
  ########### Naives Bayes .######################
subject_rdd_list=[subject1,subject2,subject3,subject4,subject5,subject6,subject7,subject8]
list_error_nb=[]
for i in range(len(subject_rdd_list)):

  subject_rdd_list=[subject1,subject2,subject3,subject4,subject5,subject6,subject7,subject8]

  test_subject=subject_rdd_list[i].map(lambda x: (map(abs, x)))

  del subject_rdd_list[i]

  dataset_subject_without_missing_value=sc.union(subject_rdd_list)
  #Load and parse the data file into an RDD of LabeledPoint.
  data = dataset_subject_without_missing_value.map(lambda x: (map(abs, x))).map(parsePoint)
  # Split the data into training and test sets (30% held out for testing)
  trainingData = data #All the subject except 1
  testData = test_subject.map(parsePoint)  #On the subject leaves

  ########## Naive Bayes model.######################
  # Train a naive Bayes model.
  model = NaiveBayes.train(trainingData, 1.0)

  # Make prediction and test accuracy.
  predictionAndLabel = testData.map(lambda p: (model.predict(p.features), p.label))
  accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x != v).count() / testData.count()
  list_error_nb.append(accuracy)

print("Average Error Random Forest ")
print reduce(lambda x, y: x + y, list_error_rf) / len(list_error_rf)

print("Average Error Logistic Regression")
print reduce(lambda x, y: x + y, list_error_lr) / len(list_error_lr)

print("Average Error Decision Tree")
print reduce(lambda x, y: x + y, list_error_dt) / len(list_error_dt)

print("Average Error Naive Bayes")
print reduce(lambda x, y: x + y, list_error_nb) / len(list_error_nb)


