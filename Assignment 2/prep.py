import re, os, xml.dom.minidom, string, bisect, pickle, json

class Node:
    """
    letter is the value of this node
    children_letters is an array which contains the letters of children nodes
    children_nodes is an array of pointers to children nodes
    is_terminal stores if this node is end of any word(last character of any word)
    """
    def __init__(self, letter,children_letters,children_nodes):
        self.letter = letter
        self.children_letters = children_letters
        self.children_nodes = children_nodes
        self.is_terminal = False

stop_words={}
punctuations = list(string.punctuation)
punc_unicodes = {ord(i): ord(' ') for i in string.punctuation}

def find(array, elem):
    i = bisect.bisect_left(array, elem)
    if i != len(array) and array[i] == elem:
        return i
    return -1

def update_posting_lists(posting_lists, documents):
    for news in documents:
        doc = xml.dom.minidom.parseString(news)
        tokenized_text = []
        
        doc_id = int(dict(doc.documentElement.attributes.items())['NEWID'])
        
        text = doc.getElementsByTagName('TEXT')
        if text:
            tokenized_text += normalize_tokenize(text[0].childNodes[0].nodeValue)
            
        title = doc.getElementsByTagName('TITLE')
        if title:
            tokenized_text += normalize_tokenize(title[0].childNodes[0].nodeValue)

        body = doc.getElementsByTagName('BODY')
        if body:
            tokenized_text += normalize_tokenize(body[0].childNodes[0].nodeValue)
            
        posting_lists[doc_id] = tokenized_text
    
    return posting_lists

def create_trie(post_list):
    all_words = set()
    for words in post_list.values():
        for word in words:
            all_words.add(word)
    
    root = Node('',[],[])
    current = root
    for word in all_words:
        current = root
        for i in range(len(word)):
            char = word[i]
            index = find(current.children_letters, char)
            if index == -1:
                pos = bisect.bisect_left(current.children_letters, char) # find the pos to insert
                bisect.insort(current.children_letters,char)
                node = Node(char,[],[])
                current.children_nodes.insert(pos,node)# insert into the same position
                current = node
            else:
                current = current.children_nodes[index]
            if i == len(word) - 1: current.is_terminal = True
   
    return root
def create_inverted_index(post_list):
    inverted_index ={}
    for words in post_list.values():
        for word in words:
            inverted_index[word] = set()

    for key in sorted(post_list.keys()):
        for word in post_list[key]:
            inverted_index[word].add(key)
    
    return inverted_index

def normalize_tokenize(text):
    text = text.replace('\n',' ')
    text = text.lower()
    text = text.translate(punc_unicodes)
    text = text.split(' ')
    text = [word for word in text if word not in stop_words]
    return set(text)

def main():
    global stop_words
    st_words = open('stopwords.txt', 'r')
    stop_words = st_words.read()
    st_words.close()
    stop_words = stop_words.split('\n')
    stop_words.append('')
    stop_words = {word:0 for word in stop_words}

    bad_char_pattern = re.compile(r"&#\d*;")
    document_pattern = re.compile(r"<REUTERS.*?<\/REUTERS>", re.S)

    posting_lists={}
    for file_name in os.listdir('reuters21578'):
        if '.sgm' not in file_name: continue

        file = open(f'reuters21578/{file_name}', 'r', encoding = 'latin-1')
        data = file.read()
        file.close()
        
        xml_data = bad_char_pattern.sub('', data)
   
        documents = document_pattern.findall(xml_data)
        posting_lists = update_posting_lists(posting_lists,documents)
    
    root = create_trie(posting_lists)
    inverted_index = create_inverted_index(posting_lists)

    tr_file = open('trie.pickle', 'ab') 
    pickle.dump(root, tr_file)                      
    tr_file.close() 
    
    inv_file = open('inverted_index.pickle','ab')
    json.dump(inverted_index, inv_file)                    
    inv_file.close()

main()