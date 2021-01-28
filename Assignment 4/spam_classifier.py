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

def get_mutual_info(train_legitimate, train_spam):
    total_doc = len(train_legitimate) + len(train_spam)
    all_words = set()

    word_idf_leg = dict() # store number of legitimate documents that contain the word for each word.
    word_idf_spam = dict() # store number of spam documents that contain the word for each word.

    for doc in train_legitimate: # create doc freq for legitimate documents
        for word in set(doc):
            all_words.add(word)
            if word in word_idf_leg:
                word_idf_leg[word] +=1
            else:
                word_idf_leg[word] = 1
    
    for doc in train_spam: # create doc freq for spam documents
        for word in set(doc):
            all_words.add(word)

            if word in word_idf_spam:
                word_idf_spam[word] +=1
            else:
                word_idf_spam[word] = 1
    
    informativeness = dict()
    
    for word in all_words:
        info = 0
        
        if word not in word_idf_spam: word_idf_spam[word] = 0 # it does not occur so add it with 0 to avoid errors
        if word not in word_idf_leg: word_idf_leg[word] = 0 # it does not occur so add it with 0 to avoid errors

        #joint probabilities
        pw_1c_1 = word_idf_spam[word] / total_doc
        pw_1c_0 = word_idf_leg[word] / total_doc
        pw_0c_1 = (len(train_spam) - word_idf_spam[word]) / total_doc
        pw_0c_0 =  (len(train_legitimate) - word_idf_leg[word]) / total_doc

        # word existence probabilities
        pw_1 = (word_idf_spam[word] + word_idf_leg[word]) / total_doc
        pw_0 = 1 - pw_1

        # class probabilities
        pc_1 = len(train_spam) / total_doc
        pc_0 = len(train_legitimate) / total_doc
       
        if pw_1c_1 != 0:
            info +=  pw_1c_1 * math.log(pw_1c_1 / (pw_1 * pc_1)) # word 1, class 1(spam)
        if pw_0c_1 != 0:
            info +=  pw_0c_1 * math.log(pw_0c_1 / (pw_0 * pc_1)) # word 0, class 1
        if pw_1c_0 != 0:
            info +=  pw_1c_0 * math.log(pw_1c_0 / (pw_1 * pc_0)) # word 1, class 0(legitimate)
        if pw_0c_0 != 0:
            info +=  pw_0c_0 * math.log(pw_0c_0 / (pw_0 * pc_0)) # word 0, class 0

        informativeness[word] = info
    
    return informativeness

def get_vocab_both_class(train_legitimate, train_spam, vocab_size = -1):
    """
    If vocab size is -1, it means use all words, otherwise use mutual information and select that many words
    """

    whole_legitimate = []
    for mail in train_legitimate:
        whole_legitimate += mail

    whole_spam = []   
    for mail in train_spam:
        whole_spam += mail

    if vocab_size == -1:    
        leg_size = len(whole_legitimate)
        spam_size = len(whole_spam)
        legitimate_occ = Counter(whole_legitimate)
        spam_occ = Counter(whole_spam)
    
        vocab = set(legitimate_occ.keys()).union(set(spam_occ.keys()))

        return leg_size, spam_size, legitimate_occ, spam_occ, vocab
    else:
        mutual_info = get_mutual_info(train_legitimate, train_spam)
        sorted_tuples = sorted(mutual_info.items(), key=lambda item: item[1], reverse = True)
        new_vocab_tuples = sorted_tuples[:vocab_size]
        new_vocab = set([x[0] for x in new_vocab_tuples])

        whole_legitimate = []
        for mail in train_legitimate:
            whole_legitimate += mail

        whole_spam = []   
        for mail in train_spam:
            whole_spam += mail
        
        filtered_whole_legitimate = [word for word in whole_legitimate if word in new_vocab]
        filtered_whole_spam = [word for word in whole_spam if word in new_vocab]

        leg_size = len(filtered_whole_legitimate)
        spam_size = len(filtered_whole_spam)
        legitimate_occ = Counter(filtered_whole_legitimate)
        spam_occ = Counter(filtered_whole_spam)
        
        return leg_size, spam_size, legitimate_occ, spam_occ, new_vocab

def mult_naive_bayes_train(train_legitimate, train_spam, vocab_size = -1, alpha = 1):
    """
    If vocab size is -1, it means use all words, otherwise use mutual information and select that many words
    """

    leg_word_probs = {}
    spam_word_probs = {}

    leg_size, spam_size, legitimate_occ, spam_occ, vocab = get_vocab_both_class(train_legitimate, train_spam, vocab_size)
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

leg_word_probs, spam_word_probs = mult_naive_bayes_train(train_legitimate, train_spam, 100)

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

