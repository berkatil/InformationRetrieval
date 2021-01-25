import sys, os, string, math
from collections import Counter

class Statistics:
    def __init__(self, prec, rec):
        self.precision = prec
        self.recall = rec
        self.F_score = (2 * prec * rec) / (prec + rec)

    def print_stat(self):
        print('Precision: ',self.precision)
        print('Recall: ',self.recall)
        print('F-measure: ',self.F_score)

punctuations = list(string.punctuation)
punctuations.append('\n')
punc_unicodes = {ord(i): ord(' ') for i in punctuations}

train_folder = sys.argv[1]
test_folder = sys.argv[2]

train_legitimate = []
train_spam = []
test_legitimate = []
test_spam = []


def pre_process(data):
    data = data.lower()
    data = data.translate(punc_unicodes)
    processed_data = []
    splitted_data = data.split(' ')
    splitted_data = [token for token in splitted_data if ((token != '') and (token != '\n'))]

    return splitted_data

def mult_naive_bayes_train(train_legitimate, train_spam, alpha = 0):
    leg_word_probs = {}
    spam_word_probs = {}

    whole_legitimate = []

    for mail in train_legitimate:
        whole_legitimate += mail

    whole_spam = []   
    for mail in train_spam:
        whole_spam += mail
    
    leg_size = len(whole_legitimate)
    spam_size = len(whole_spam)
    legitimate_occ = Counter(whole_legitimate)
    spam_occ = Counter(whole_spam)

    vocab = set(legitimate_occ.keys()).union(set(spam_occ.keys()))
    vocab_size = len(vocab)

    for token in vocab:
        leg_word_probs[token] = (legitimate_occ[token] + alpha) / (leg_size + alpha * vocab_size) # prob with laplace smoothing
        spam_word_probs[token] = (spam_occ[token] + alpha) / (spam_size + alpha * vocab_size)

    return leg_word_probs, spam_word_probs

def evaluate(legitimate_labels, spam_labels):
    # positive means spam
    TP = sum(spam_labels) # number of ones in spam labels
    FP = sum(legitimate_labels) # number of ones in legitimate labels
    TN = len(legitimate_labels) - sum(legitimate_labels)
    FN = len(spam_labels) - sum(spam_labels)
    
    leg_stat = Statistics(TN / (TN + FN) , TN / (TN + FP))    
    spam_stat = Statistics(TP / (TP + FP) , TP / (TP + FN))

    return leg_stat, spam_stat

def calculate_prob(document, word_probs):
    log_prob = math.log(0.5) # class prob we have equally sized data so it is 0.5
    
    for token in document:
        if token in word_probs:
            log_prob += math.log(word_probs[token])
        else:
            log_prob += -100000 # this token is not in our train set so penalize
    
    return log_prob
    
def mult_naive_bayes_test(test_legitimate, test_spam, leg_word_probs, spam_word_probs):
    legitimate_labels = []
    spam_labels = []
    
    for mail in test_legitimate:
        legitimate_prob = calculate_prob(mail, leg_word_probs)
        spam_prob = calculate_prob(mail, spam_word_probs)

        if legitimate_prob > spam_prob:
            legitimate_labels.append(0) # not spam
        else:
            legitimate_labels.append(1) # spam

    for mail in test_spam:
        legitimate_prob = calculate_prob(mail, leg_word_probs)
        spam_prob = calculate_prob(mail, spam_word_probs)

        if legitimate_prob > spam_prob:
            spam_labels.append(0) # not spam
        else:
            spam_labels.append(1) # spam
    
    return legitimate_labels, spam_labels

for file_path in os.listdir(f'{train_folder}/legitimate'):
    if file_path.startswith('.'): continue # discard hidden file
    f = open(f'{train_folder}/legitimate/{file_path}', "r",encoding = 'latin-1')
    data = f.read()
    f.close()
    data = data[8:] # discard Subject:
    train_legitimate.append(pre_process(data))

for file_path in os.listdir(f'{train_folder}/spam'):
    if file_path.startswith('.'): continue
    f = open(f'{train_folder}/spam/{file_path}', "r",encoding = 'latin-1')
    data = f.read()
    f.close()
    data = data[8:] # discard Subject:
    train_spam.append(pre_process(data))

for file_path in os.listdir(f'{test_folder}/legitimate'):
    if file_path.startswith('.'): continue
    f = open(f'{test_folder}/legitimate/{file_path}', "r",encoding = 'latin-1')
    data = f.read()
    f.close()
    data = data[8:] # discard Subject:
    test_legitimate.append(pre_process(data))

for file_path in os.listdir(f'{test_folder}/spam'):
    if file_path.startswith('.'): continue
    f = open(f'{test_folder}/spam/{file_path}', "r",encoding = 'latin-1')
    data = f.read()
    f.close()
    data = data[8:] # discard Subject:
    test_spam.append(pre_process(data))

leg_word_probs, spam_word_probs = mult_naive_bayes_train(train_legitimate, train_spam, 1)

legitimate_labels, spam_labels = mult_naive_bayes_test(test_legitimate, test_spam, leg_word_probs, spam_word_probs)

leg_stat, spam_stat = evaluate(legitimate_labels, spam_labels)

print("Legitimate Class")
leg_stat.print_stat()
print("\nSpam Class")
spam_stat.print_stat()

print("\nMacro Average Scores")
print('Precision: ', (leg_stat.precision + spam_stat.precision) / 2)
print('Recall: ', (leg_stat.recall + spam_stat.recall) / 2)
print('F-measure: ', (leg_stat.F_score + spam_stat.F_score) / 2)

