import pandas as pd
import urllib.request
from pororo import Pororo

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")
test = pd.read_table('ratings_test.txt')

sa = Pororo(task="sentiment", model="brainbert.base.ko.nsmc", lang="ko")
zsl = Pororo(task="zero-topic", lang="ko")

def get_accuracy_binary(sent,mode): # Positive/Negative
    X = sent.document.values
    y = sent.label.values
    y_pred = []
    if mode =='zsl':
        for line in X:
            val = zsl(line,['긍정','부정'])
            max_key = max(val.items(), key=operator.itemgetter(1))[0]
            if max_key =='긍정':
                y_pred.append(1)
            else:
                y_pred.append(0)
    elif mode=='sa:
        for line in X:
            val = sa(line)
            if val == 'Negative':
                y_pred.append(0)
            else:
                y_pred.append(1)
    return y,y_pred
  
if __name__ == '__main__':
  test_size = 1000
  y,zsl_pred = get_accuracy_binary(test[:test_size],'zsl')
  y,sa_pred = get_accuracy_binary(test[:test_size],'sa')

  print(f'Zero Shot Topic Classification Accuracy {np.equal(y,zsl_pred).sum()/test_size*100}')
  print(f'Sentiment Analysis Accuracy {np.equal(y,sa_pred).sum()/test_size*100}')
