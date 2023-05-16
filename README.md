# wine-review-prediction
Predicting Wine Reviewer Name using Wine Reviews

Every day, dozens of wines are reviewed by professional reviewers. They write up short descriptions of the wines they taste and assign a score out of 100 to indicate quality. Certain reviewers build up a reputation in the wine industry and gather a dedicated following of fans. These fans rely on their favorite reviewer to know which wines to try.
This begs the question: are there distinguishing features in the language that reviewers use to describe wines? Can we use the information contained within wine reviews to predict which reviewer they have been written by?


Dataset
Dataset consist of roughly 130,000 wine reviews that have been scraped from www.winemag.com, the official website of the publication ‘Wine Enthusiast’:

Approach
First step was to create baseline model with following steps.
Exploratory data analysis- Understand Data distribution,  check if target labels are balanced/not, Type-Supervised classification problem, Checking for na’s.
Data Cleaning –fixing na’s, removing outliers.
Feature Engineering-filtering columns, Label encoding, converting datasets into Electra compatible format.
Modeling- As we know transformer models have achieved State of art results when it comes to text classification. Upon reviewing model performances on publicly available data & evaluating GLUE benchmark, (Electra, XLNet, ALBERT models were chosen for this task).
Electra was the first choice as its lighter in weight which makes it faster and computationally less expensive. 
Electra was finetuned on given dataset and resulted in accuracy of 96%.

Future work
Train on ELECTRA-Large
Since data has a good amount of missing values, we can use the ELECTRA model that we trained to predict the missing target values(taster names) and retrain the model on whole dataset(this will give better accuracy)
Take a general intro about the taster to understand his tone and preferences and incorporate as a feature in our model 
Add review limit <512 to ensure we don’t cut out reviews as some important comments might be at the end which are truncated by model
To incorporate longer reviews we can try Longformer transformer model
We can use other features like title and check performance (this will require separate tokenization and cleaning steps)





