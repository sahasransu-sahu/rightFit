import os

import docx2txt
import gensim
import nltk
from gensim.corpora import TextCorpus
from gensim.similarities import Similarity

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile, datapath
from nltk.tokenize import word_tokenize, sent_tokenize


def matchPercent(job_description, resume):
    text = [resume, job_description]

    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text)

    # get the match percentage
    matchPercentage = cosine_similarity(count_matrix)[0][1] * 100
    matchPercentage = round(matchPercentage, 2)  # round to two decimal
    return matchPercentage


def nResumetoJob():
    resume = docx2txt.process("SAHASRANSU_SAHU_LATEST_RESUME.docx")
    resume2 = docx2txt.process("resume.docx")
    #print(resume.replace('\n', ' ').replace('\t', ' '))
    print('=====================================================================================')

    # Store the job description into a variable
    job_description = docx2txt.process("job_description.docx")
    print(job_description)
    print('=====================================================================================')


    print("Your resume matches about " + str(matchPercent(job_description, resume)) + "% of the job description.")
    print('=====================================================================================')

    print("Your resume2 matches about " + str(matchPercent(job_description, resume2)) + "% of the job description.")
    print('=====================================================================================')



def nJobstoResume() :
    file1 = "SAHASRANSU_SAHU_LATEST_RESUME.docx"

    resume = docx2txt.process("SAHASRANSU_SAHU_LATEST_RESUME.docx")
    print(resume.replace('\n', ' ').replace('\t', ' '))
    print('=====================================================================================')

    # Store the job description into a variable
    job_description = docx2txt.process("job_description.docx")
    #print(job_description)
    #print('=====================================================================================')

    job_description2 = docx2txt.process("dummyDesc.docx")
    #print(job_description)
    #print('=====================================================================================')



    print("Your resume matches about " + str(matchPercent(job_description, resume)) + "% of the job description.")
    print('=====================================================================================')

    print("Your resume matches about " + str(matchPercent(job_description2, resume)) + "% of the job description2.")
    print('=====================================================================================')


def prepareFolder(description_file_path):
    files_docs=[]
    '''added line '''
    for filename in sorted(os.listdir(description_file_path)):
        if filename.endswith('.txt'):
            file_content=''
            filepath=description_file_path+'/'+filename
            with open(filepath) as f:
                tokens = sent_tokenize(f.read())
                for line in tokens:
                    file_content=file_content+ line
            files_docs.append(file_content)
            #print(file_content)
        # added line
    return files_docs,sorted(os.listdir(description_file_path))


def prepareFile(file_path):
    resumeContent = []
    file_content = ''
    with open(file_path) as f:

        tokens = sent_tokenize(f.read())
        for line in tokens:
            file_content = file_content + line

    resumeContent.append(file_content)
    #print (resumeContent)
    return resumeContent

def nJobtoResumeGensim():
    descriptions, file_names= prepareFolder('job_desciptions')  #argument is path to jd folders
    print("Number of Job descriptions:", len(descriptions))
    #print (descriptions)

    '''creating index for descriptions'''
    gen_docs = [[w.lower() for w in word_tokenize(text)] for text in descriptions]
    dictionary = gensim.corpora.Dictionary(gen_docs)
    # print(dictionary)
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    # print(corpus)
    tf_idf = gensim.models.TfidfModel(corpus)
    sims = gensim.similarities.Similarity(None, tf_idf[corpus], num_features=len(dictionary))

    '''get resume content'''
    resumeContent = prepareFile('SAHASRANSU_SAHU_LATEST_RESUME.txt')
    for line in resumeContent:
        query_doc = [w.lower() for w in word_tokenize(line)]
        query_doc_bow = dictionary.doc2bow(query_doc)


    query_doc_tf_idf = tf_idf[query_doc_bow]
    print('Comparing Result:', sims[query_doc_tf_idf])
    print(file_names)

def nResumetoJobGensim():
    descriptions,filenames = prepareFolder('resumes')  #argument is path to resume folders
    print("Number of Resumes :", len(descriptions))
    #print (descriptions)

    '''creating index for resumes'''
    gen_docs = [[w.lower() for w in word_tokenize(text)] for text in descriptions]
    dictionary = gensim.corpora.Dictionary(gen_docs)
    # print(dictionary)
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    # print(corpus)
    tf_idf = gensim.models.TfidfModel(corpus)
    sims = gensim.similarities.Similarity(None, tf_idf[corpus], num_features=len(dictionary))


    '''get job content'''
    resumeContent= prepareFile('job_description.txt')
    for line in resumeContent:
        query_doc = [w.lower() for w in word_tokenize(line)]
        query_doc_bow = dictionary.doc2bow(query_doc)

    query_doc_tf_idf = tf_idf[query_doc_bow]
    print('Comparing Result:', sims[query_doc_tf_idf])
    print('corresponding  f:', filenames)




if __name__ == "__main__":
    print('=====================================================================================')
    print('===================Multiple Jobs, one Resume=========================================')
    nJobtoResumeGensim()
    print('=====================================================================================')
    print('=====================================================================================')
    print('===================Multiple Resumes, one Job=========================================')
    nResumetoJobGensim()
    print('=====================================================================================')
