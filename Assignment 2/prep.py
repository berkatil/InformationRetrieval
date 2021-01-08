import re, os, xml.dom.minidom, string, bisect, pickle, json, sys
import Trie

stop_words={}
punctuations = list(string.punctuation)
punc_unicodes = {ord(i): ord(' ') for i in string.punctuation}

def update_posting_lists(posting_lists, documents):
    for news in documents:
        doc = xml.dom.minidom.parseString(news)
        tokenized_text = []
        
        doc_id = int(dict(doc.documentElement.attributes.items())['NEWID'])
            
        title = doc.getElementsByTagName('TITLE')
        if title:
            tokenized_text += normalize_tokenize(title[0].childNodes[0].nodeValue)

        body = doc.getElementsByTagName('BODY')
        if body:
            tokenized_text += normalize_tokenize(body[0].childNodes[0].nodeValue)
            
        posting_lists[doc_id] = tokenized_text
    
    return posting_lists

def create_trie(post_list):# post_list is a map whose keys are document IDs and values are the words in that document
    all_words = set()
    for words in post_list.values():
        for word in words:
            all_words.add(word)
    
    root = Trie.Node('',[],[])
    for word in all_words:
        current = root
        for i in range(len(word)):
            char = word[i]
            index = Trie.find(current.children_letters, char)# check if the children of our current node contains this char
            if index == -1:# this sequence has not been added so we need to add this char into a child node
                pos = bisect.bisect_left(current.children_letters, char) # find the position to insert
                bisect.insort(current.children_letters,char)
                node = Trie.Node(char,[],[])
                current.children_nodes.insert(pos,node)# insert into the same position
                current = node
            else:# this sequence is already added so move to that node
                current = current.children_nodes[index]
            if i == len(word) - 1: current.is_terminal = True
   
    return root

def create_inverted_index(post_list):
    inverted_index ={}
    for words in post_list.values():
        for word in words:
            inverted_index[word] = set()

    for key in post_list.keys():
        for word in post_list[key]:
            inverted_index[word].add(key)
    
    for key in inverted_index.keys():
        inverted_index[key] = list(inverted_index[key])

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

    remove_pattern = re.compile(r"&#\d*;")
    document_pattern = re.compile(r"<REUTERS.*?<\/REUTERS>", re.S)

    posting_lists={}

    data_folder = sys.argv[1]
    for file_name in os.listdir(data_folder):
        if '.sgm' not in file_name: continue

        file = open(f'{data_folder}/{file_name}', 'r', encoding = 'latin-1')
        data = file.read()
        file.close()
        
        xml_data = remove_pattern.sub('', data)
        
        documents = document_pattern.findall(xml_data)
        posting_lists = update_posting_lists(posting_lists,documents)
        
    root = create_trie(posting_lists)
    inverted_index = create_inverted_index(posting_lists)

    tr_file = open('trie.pickle', 'ab') 
    pickle.dump(root, tr_file)                      
    tr_file.close() 
   
    inv_file = open('inverted_index.json','w')
    json.dump(inverted_index, inv_file)                    
    inv_file.close()
   
main()
