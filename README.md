## CourseWork of MachineLearning 2021 in EE, THU

### Predict of RoadDrive

This is an implementation of road predict task on kaggle. [web-page](https://www.kaggle.com/absolutegaming/road-prediction)
Here we use multi-layer logistic regression to predict road situation, and use GDA to predict driver status.

Running task

```shell
cd RoadDrive
python roadJudge.py
```

### Predict of Employee

This task is about predicting whether a employee would leave the company in next 2 years. [web-page](https://www.kaggle.com/tejashvi14/employee-future-prediction)
Here we use random forest to complete the task. Each tree is an ID3 tree.
Running task

```shell
cd EmployeePredict
python EmployeePredict.py --trees 10
```

### Spam Classification

This task is spam classification using the SMS Spam Collection Dataset. [web-page](https://www.kaggle.com/uciml/sms-spam-collection-dataset)
We implement a LSTM RNN to accomplish this task.
