import requests,re, string, math, sys, pickle, json
from collections import Counter

punctuations = list(string.punctuation)
punc_unicodes = {ord(i): ord(' ') for i in string.punctuation}

def cosine_similarity(vector1,vector2): # to calculate cosine similarity between 2 vectors
    nominator = sum(p1 * p2 for p1, p2 in zip(vector1,vector2)) # dot product

    length1 = math.sqrt(sum(p*p for p in vector1))
    length2 = math.sqrt(sum(p*p for p in vector2))
    if length1 == 0: length1 = 0.0000000000001
    if length2 == 0: length2 = 0.0000000000001
    return nominator/(length1 * length2)

def get_title(data):
    return re.findall(r'<h1 id="bookTitle".*?>(.*?)</h1>',data.text,re.DOTALL)[0].strip()

def get_authors(data):
    authors = re.findall(r'div id="bookAuthors" class="">(.*?)<div id',data.text,re.DOTALL)[0]
    authors = re.findall(r'<div class=\'authorName__container\'>(.*?)</div>',authors,re.DOTALL)

    author_names = []

    for author in authors:
        author_name = re.findall(r'<span itemprop="name".*?>(.*?)</span>',author,re.DOTALL)[0]
        if 'span class="greyText"' in author:
            author_name += re.findall(r'<span class="greyText.*?">(.*?)</span>',author,re.DOTALL)[0]

        author_names.append(author_name)
   
    return ','.join(author_names)

def get_description(data):
    return re.findall(r"descriptionContainer.*?freeTextContainer.*?freeText.*?>(.*?)</span>\n", data.text,re.DOTALL)[0].strip()
   
def get_rec_book_urls(data):
    rec_books = re.findall(r'<li class=\'cover\' id=\'bookCover.*?href="(.*?)">',data.text,re.DOTALL)
    return rec_books

def get_genres(data):
    genres = re.findall(r'class="actionLinkLite bookPageGenreLink" href=.*?>(.*?)</a>',data.text,re.DOTALL)
    return [genre.lower() for genre in genres]

def pre_process(document):
    text = document.replace('\n',' ')
    text = text.lower()
    text = text.translate(punc_unicodes) # discard punctuations
    return text.split(' ')

def select_informative_words(all_words_occurences, corpus_size):#takes a dict which contains all of the words and their document frequencies
    min_threshold = int(0.006*corpus_size)
    max_threshold = int(0.9*corpus_size)
    
    vocabulary = dict()

    for key in all_words_occurences:
        if (all_words_occurences[key] < max_threshold) and (all_words_occurences[key] > min_threshold):
            vocabulary[key] = all_words_occurences[key]
    
    return vocabulary

def recommend(book_url, book_tfidf, book_genre, all_tfidfs, all_genre_vectors, index_url, alfa):
    cos_similiarities = []

    for index in range(len(all_tfidfs)):
        desc_sim = cosine_similarity(book_tfidf, all_tfidfs[index])
        genre_sim = cosine_similarity(book_genre, all_genre_vectors[index])

        cos_similiarities.append((index, alfa*desc_sim + (1-alfa)*genre_sim))

    sorted_cos_sim = sorted(cos_similiarities,key = lambda x: x[1],reverse=True)# sort documents according to cosine similarities

    if len(sorted_cos_sim) >= 19 :
        pot_recommendations = sorted_cos_sim[:19]# get 19 in case of the first one is the url itself    
    else:
        pot_recommendations = sorted_cos_sim

    first_index = str(pot_recommendations[0][0])
   
    if (first_index in index_url) and (book_url == index_url[first_index]):
        pot_recommendations = pot_recommendations[1:]
    elif len(sorted_cos_sim) >= 18: pot_recommendations = pot_recommendations[:18]

    rec_urls = [index_url[str(rec[0])] for rec in pot_recommendations]
    
    return rec_urls

def single_book_genre(all_genres, doc): # create genre document for a single url
    bin_term_vector = []

    for genre in all_genres:
        if genre in doc:
            bin_term_vector.append(1)
        else:
            bin_term_vector.append(0)

    return bin_term_vector

def single_book_tfidf(vocabulary, all_words_occurences, doc, N): # create description for a single url
    occurences = Counter(doc)
    tf_vector = []
    idf_vector = []

    for word in vocabulary:
        if word in occurences:
            tf_vector.append(1 + math.log10(occurences[word]))
        else:
            tf_vector.append(1)
        
        idf_vector.append(math.log10(N / all_words_occurences[word]))
    
    return [tf * idf for tf, idf in zip(tf_vector, idf_vector)]

def precision(recommendations,ground_truth):
    tp = 0
    for url in recommendations:
        if url in ground_truth:
            tp += 1
    
    return tp/len(recommendations)

def evaluate(recommendations, ground_truth):
    prec = precision(recommendations,ground_truth)
    total_relevant = 0
    total_prec = 0
    for index in range(len(recommendations)): # to calculate average precision
        if(recommendations[index] in ground_truth):
            total_relevant += 1
            total_prec += precision(recommendations[:index+1],ground_truth)
    
    avg_prec = 0
    if total_relevant !=0 : avg_prec = total_prec/total_relevant

    return prec, avg_prec

def get_tfidf_vectors(documents):# list of list each element is a document containng only tokens
    all_words_occurences = dict() # store number of documents that contain the word for each word. This is for IDF
    
    for doc in documents: # create document frequencies
        for word in set(doc):
            if word in all_words_occurences:
                all_words_occurences[word] +=1
            else:
                all_words_occurences[word] = 1

    vocabulary = select_informative_words(all_words_occurences, len(documents)) # discard some words

    N = len(documents)

    tf_idf_vectors = []

    voc_file = open('vocabulary.json', 'w') 
    json.dump(vocabulary, voc_file)                      
    voc_file.close()

    all_word_file = open('all_word_occurences.json', 'w') 
    json.dump(all_words_occurences, all_word_file)                      
    all_word_file.close()
    
    for doc in documents:# calculate tf-idf vectors
        tf_vector = []
        idf_vector = []
        occurences = Counter(doc)

        for word in vocabulary:
            if word in occurences:
                tf_vector.append(1 + math.log10(occurences[word]))
            else:
                tf_vector.append(0)
            
            idf_vector.append(math.log10(N / all_words_occurences[word]))

        tf_idf_vectors.append([tf * idf for tf, idf in zip(tf_vector, idf_vector)])
    
    return tf_idf_vectors

def get_genre_vectors(documents):
    all_genres = set()

    for genres in documents:
        for genre in genres:
            all_genres.add(genre)

    all_genres = list(all_genres)

    bin_term_vectors = []

    genre_file = open('all_genres.pickle', 'ab') 
    pickle.dump(all_genres, genre_file)                      
    genre_file.close()

    for doc in documents:# create genre vectors
        bin_term_vector = []

        for genre in all_genres:
            if genre in doc:
                bin_term_vector.append(1)
            else:
                bin_term_vector.append(0)
    
        bin_term_vectors.append(bin_term_vector)
    
    return bin_term_vectors

def get_documents(file_path='', url = ''):
    if file_path != '':# full pipeline
        fp = open(file_path, 'r') 
        index_url = dict()
        index = 0
        url_title_author = dict()
        descriptions = []
        genres = []
        
        for line in fp:
            line = line.strip()
            number_of_tries = 0
            is_retrieved = False
            while(not(is_retrieved) and (number_of_tries != 3)):
                try:
                    data = requests.get(line)
                    authors = get_authors(data)
                    description = get_description(data)
                    title = get_title(data)
                    genre = get_genres(data)

                    genres.append(genre)    
                    descriptions.append(pre_process(description))
                    url_title_author[line] = f'{title} -> {authors}'
                    index_url[index] = line
                    index += 1
                    is_retrieved = True
                
                except:
                    print(line,' cannot retrieved')
                    number_of_tries += 1

        tf_idf_vectors = get_tfidf_vectors(descriptions)
        genre_vectors = get_genre_vectors(genres)

        doc_file = open('document_vectors.pickle', 'ab') #tfidf for each doc
        pickle.dump(tf_idf_vectors, doc_file)                      
        doc_file.close()

        genre_file = open('genre_vectors.pickle', 'ab') # genre vector for each doc
        pickle.dump(genre_vectors, genre_file)                      
        genre_file.close()

        url_index_file = open('index_url.json','w') # index->url dictionary
        json.dump(index_url, url_index_file)                    
        url_index_file.close() 

        url_title_file = open('url_title_author.json','w') # url-> title-author dictionary
        json.dump(url_title_author, url_title_file)                    
        url_title_file.close()

        descs = open('descriptions.pickle', 'ab') #descriptions
        pickle.dump(descriptions, descs)                      
        descs.close()

        genre_content = open('genres.pickle', 'ab') #genres
        pickle.dump(genres, genre_content)                      
        genre_content.close()

    elif url != '': # 1 book so recommend book and make an evaluation 
        doc_file = open('document_vectors.pickle', 'rb') # tfidfs
        documents =  pickle.load(doc_file)
        doc_file.close()
        
        genre_file = open('genre_vectors.pickle', 'rb') # genre vectors
        genre_vectors = pickle.load(genre_file)                      
        genre_file.close()
       
        url_index_file = open('index_url.json','rb') # index->url dictionary
        index_url = json.load(url_index_file)                    
        url_index_file.close()
        
        url_title_file = open('url_title_author.json','rb') # url-> title-author dictionary
        url_title_author = json.load(url_title_file)                    
        url_title_file.close()
       
        voc_file = open('vocabulary.json', 'rb') # all terms in the corpus
        vocabulary = json.load(voc_file)                      
        voc_file.close()
        
        all_word_file = open('all_word_occurences.json', 'rb') # inverse term freq
        all_words_occurences = json.load(all_word_file)                      
        all_word_file.close()
        
        genre_file = open('all_genres.pickle', 'rb') #vocab for all genres
        all_genres = pickle.load(genre_file)                      
        genre_file.close()
       
        data = requests.get(url)
        
        authors = get_authors(data)
        description = get_description(data)
        title = get_title(data)
        genre = get_genres(data)

        print('Title of the book: ',title)
        print('Description of the book: ',description)
        print('Author of the book: ',authors)
        print('Genres of the book: ',genre)

        ground_truths = get_rec_book_urls(data)

        genre_vector = single_book_genre(all_genres, genre)
        tf_idf_vector = single_book_tfidf(vocabulary, all_words_occurences, description, len(url_title_author))
        recommended_urls = recommend(url,tf_idf_vector, genre_vector,documents,genre_vectors,index_url,0.5)
        prec, AP = evaluate(recommended_urls, ground_truths)

        print('Recommendations')
        for url in recommended_urls:
            print(url_title_author[url])

        print('Precision',prec)
        print('Average Precision',AP)
        return prec,AP
    else:
        print('There is something wrong')


filepath = sys.argv[1]
if '.txt' in filepath:
    get_documents(file_path = filepath)
else:
    get_documents(url=filepath)
