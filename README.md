# IBM_Call_for_code
Data collection and clean process

The project's purpose is to identify whether the black people suffer the unequal treatment in sentencing process and they may have more serious charges than other races. 

Basically, in order to build better model, we need to collect data from different states and counties. The following link is the open sentencing data from Cook County's Government's website. https://datacatalog.cookcountyil.gov/Courts/Sentencing/tg8v-tm6u/data.

To be specific, we mainly want the attributes including races, age, date, the charge, incident, length of case in days, sentence type, etc. With those data, we could change the race from black to other races and run the model to see whether the model have the same predicitons. According to FOIA, we could require the data from government. Hence, my team member Tyler would be responsible for this issue and the following part is the template email.

Hello,
My name is *****, and I am an open source contributor for IBM's Call for Code Open Sentencing project. My team and I aim to mitigate racial bias in the judicial system with an automated reasoner designed to identify racial discrepancies between the ordinary sentence length for a given offense and the sentence assigned by the judge of a particular case. We are currently in need of datasets to train our model, and after examining the large .mdb file provided in the FDC's records requests page, we found the data too general to be of use to our model. Therefore, we would like to request any and all ethnicity-specific data that the state of Florida can provide, where datasets enumerating individual cases by offense, ethnicity of offender, and sentence length would be ideal. Furthermore, attached is a sample .csv file from Cook County, Illinois with case-specific data, and any datasets of a similar format and with similar statistics would be ideal.
If there is a process by which we need to obtain this data, or if there are any other specifics that we may offer to accelerate the process of acquiring data, please let us know. Any assistance is greatly appreciated.

Regards,
IBM Call for Code Open Sentencing Team

1. Beginning State:
At the beginning state, I mainly collect data from two resources, the New York State government and Florida Department of Corrections.
For New York State Government:
we have two requirments. Firstly, we want the excel form of the following link.
https://www.criminaljustice.ny.gov/crimnet/ojsa/comparison-population-arrests-prison-demographics.html

And also, we want the specific data like the cook county's data.

For Florida Department of Corrections:
http://www.dc.state.fl.us/pub/obis_request.html
They provides the data base link as above which denotes the information of criminals in the prison. This data base is great but it does not contain much useful information for our project. Hence, we want more attributes including the date criminals are put into jain and the date they are released. And also, we need to know their race, age, number of days they stayed in jail. We have sent emails to them for further response and more specific data.

2. Second state:
Since I haven't received any dataset from the government, I am trying to use the linear regression to do the validations. To be specific, I would separate the data into two parts, the black and the other races. And then, we build the linear regression model independently and train the data and calculate out the weight. Then, I would like to see how the model output based on different weights. 

Due to the number of the data (256k entities), it would take a very long time to train the data. Hence, the model is still building now.
