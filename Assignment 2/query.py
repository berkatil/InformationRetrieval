import pickle, json, bisect, sys, Trie

def find_words(node, pot_word):
    """
    Parameters
    ----------
    pot_word:
        A word created up to this node and contains node.letter
    """
    words = []
    if node.is_terminal: words.append(pot_word) # if we found a word add it into the list
    for child in node.children_nodes:
        words.extend(find_words(child,pot_word+child.letter))# continue to find other words

    return words

def retrieve_words(query,root,inverted_index):
    if query[-1] != '*': #check if this is a wildcard query
        if query in inverted_index.keys(): return [query] # this is a word query so if it exists in the vocabulary return itself
        else: return []

    words=[]
    current = root
    exists = True
    for i in range(len(query) - 1):# to check if starting string of the wildcard query
        index = Trie.find(current.children_letters,query[i])
        if index == -1:
            exists = False
            break
        current = current.children_nodes[index]
    
    if(not(exists)): return words
    
    words = find_words(current,query[:-1])# to find words staring with string given in the wildcard query

    return words

def list_documents(words, inverted_index):
    docs = set()
    for word in words:
        docs.update(inverted_index[word])
    print(sorted(list(docs)))

def main():
    tr_file = open('trie.pickle', 'rb')
    root = pickle.load(tr_file)
    tr_file.close()

    inv_file = open('inverted_index.json',) 
    inverted_index = json.load(inv_file)
    inv_file.close()

    query = sys.argv[1]
    words = retrieve_words(query,root,inverted_index)
    list_documents(words,inverted_index)
    
main()
