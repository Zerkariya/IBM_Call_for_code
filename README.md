# IBM_Call_for_code
#### 1. Data Source
Basically, there are two links for us to understand the dataset.
Here is the data source:
https://datacatalog.cookcountyil.gov/Courts/Sentencing/tg8v-tm6u/data

The following one is the explanation of each attribute in the dataset.
https://www.cookcountystatesattorney.org/sites/default/files/files/documents/column_by_dataset_glossary_final_1.pdf

#### 2 Date
#### 2.1 The data analyzing process is separated into three main process.
Firstly, choosing 16th attribute "charge disposition" as the y_response that we want to predict. It represent how serious the person is charged. 
And, then, the rest of data is considered as the predictors. To be clear, the information about dates, position, ect, are removed since they are not revelant. And also, the gender is also removed.

#### 2.2 Data Cleaning part
The following attributes are the arrtibutes I removed from the dataset for several reasons.
1) Some of them are useless (like case ide, case participant id....)
2) Some of them have lots of blank cells and missed values (like incident end date)

0 case id  
1 case participant id  
2 received date  
5 charge id  
6 charge version id  
9 disposition date  
10 disposition charged chapter  
11 disposition charged act  
12 disposition charged section  
14 disposition charged aoic  
16 charge disposition reason  
19 sentence court facility  
20 sentence phase  
21 sentence date  
24 commitment type  
32 incident begin date  
33 incident end date  
34 law enforcement agency  
35 law enforcement unit  
36 arrest date  
37 felony review date  
38 felony review result  
39 arraignment date  

#### 2.3 We only want the primary charge flag and current sentence flag as true and remove all other false cases

#### 2.4 All of the cases with blank cells are removed

#### 2.5 For disposition charged class, we only want M, X, 1, 2, 3, 4, A, B, and C. All the others are removed

#### 2.6 For commitment unit class, we remove the cases in "term", "dollars", and "pounds".


#### 3.1 Further, the data is separated into three subdataset. 
The first dataset only includes the people with race as Black. The second dataset only includes the people with race with no Black people. 
  
#### 3.2 Then, we run the knn and spectural on first two datasets individually and generate two weights individually. 
Hence, what we want to check is whether they have the similar predictions on the third dataset.
If there is a huge difference between their predictions, we would have two more questions to argue. 
1) Which w has higher accuracy on the third dataset's predictions? 
2) Which w produces more serious charges?


#### 4. Improvements:
To be clear, there are three files I wrote to cluster the cases. The first one is the knn which represents the k-nearest neighbours algo. The second one is Support Vector Machine. The third one is the Spectural Algo. Once you finish the clean data part, you can call those files and use them directly. 


