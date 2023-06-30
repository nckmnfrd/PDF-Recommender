import os
import PyPDF2
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_pdfs():
    directory = '/Users/nicholasmanfredi/PycharmProjects/Northwell_Search_Engine/PDFs'
    pdf_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'rb') as f:
                title = filename
                pdf_list.append(title)
    return pdf_list

def read_pdf_content():
    directory = '/Users/nicholasmanfredi/PycharmProjects/Northwell_Search_Engine/PDFs'
    pdf_content_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                title = filename
                content = ''
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.getPage(page_num)
                    content += page.extractText()

                pdf_content_list.append(content)
    return pdf_content_list


def clean_and_toke_pdf():
    lst = read_pdf_content()
    #print(lst)
    concat_lst = []
    for content in lst:
        tokenized_text = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(content)]
        #print(tokenized_text)
        concatenated_list = list(itertools.chain(*tokenized_text))
        concat_lst.append(concatenated_list)

    to_remove = ['@', ',', '.', '(', ')', ':', ';', '%']
    stop_words = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its',
                  'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with']
    remove_case = to_remove + stop_words
    cleaned_lst = []

    for lst in concat_lst:
        cleaned_sublst = []
        for word in lst:
            if word.lower() not in remove_case:
                cleaned_sublst.append(word.lower())
        cleaned_lst.append(cleaned_sublst)

    return cleaned_lst

def pdf_dictionary():
    dictionary = dict(zip(read_pdfs(), clean_and_toke_pdf()))
    return dictionary

def get_most_similar_key(input_text):
    dictionary = pdf_dictionary()
    vectorizer = TfidfVectorizer()
    corpus = [' '.join(text) for text in dictionary.values()]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    input_vector = vectorizer.transform([' '.join(nltk.word_tokenize(input_text.lower()))])
    cosine_similarities = cosine_similarity(input_vector, tfidf_matrix)
    most_similar_index = cosine_similarities.argmax()
    most_similar_key = list(dictionary.keys())[most_similar_index]
    return most_similar_key



#pdf = read_pdfs()
#print(pdf)
#pdf_content = read_pdf_content()
#print(pdf_content)
print('hello')
#print(pdf_dictionary())
#print(clean_and_toke_pdf())
#print(lemmatize_lst())
#print(pdf_dictionary())

print(get_most_similar_key('I want to learn about trading indicators'))
