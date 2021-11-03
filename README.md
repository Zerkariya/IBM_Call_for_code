# IBM_Call_for_code
1.
Basically, there are two links for us to understand the dataset.
Here is the data source:
https://datacatalog.cookcountyil.gov/Courts/Sentencing/tg8v-tm6u/data

The following one is the explanation of each attribute in the dataset.
https://www.cookcountystatesattorney.org/sites/default/files/files/documents/column_by_dataset_glossary_final_1.pdf

2.
2.1 The data analyzing process is separated into three main process.
Firstly, choosing 14th attribute DISPOSITION_CHARGED_CLASS as the y_response that we want to predict. It represent how serious the person is charged. 
And, then, the rest of data is considered as the predictors. To be clear, the information about dates, position, ect, are removed since they are not revelant. And also, the gender is also removed.

2.2 Further, the data is separated into three subdataset. The first dataset only includes the people with race as Black. The second dataset only includes the people with race with no Black people. 
The third dataset is used for the prediction comparasion part. For the first and second dataset, we separate them into three parts again as training, validation, and testing dataset. 
  
2.3 Then, we run the Ridge Regression on first two datasets individually and generate two weights individually. Hence, what we want to check is whether they have the similar predictions on the third dataset.
If there is a huge difference between their predictions, we would have two more questions to argue. 
1) Which w has higher accuracy on the third dataset's predictions? 
2) Which w produces more serious charges?

3.
Improvemens:
Since the linear regression really costs lots of time and the huge number to calculate consumes memory, we can use the Gaussian Kernel Matrix trick to make the calculation times smaller. Hence, further, I would try SVM and kernel regressio to produce the predictions and compute the graphs.


