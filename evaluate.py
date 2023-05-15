
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import numpy as np

#Evaluate
def evaluate(trainer,test_dataset,testing_data):
    model_result = trainer.evaluate()
    print('Eval accuracy',model_result['eval_acc'])
    y_pred=trainer.predict(test_dataset)
    predicted_label=np.argmax(y_pred.predictions,axis=1)

    y_test=testing_data['taster_label']
    print('F1 score:',f1_score(y_test, predicted_label, average="weighted"))
    print('Precision:',precision_score(y_test, predicted_label, average="weighted"))
    print('Recall:',recall_score(y_test, predicted_label, average="weighted")) 
    print('Accuracy:',accuracy_score(y_test,predicted_label))