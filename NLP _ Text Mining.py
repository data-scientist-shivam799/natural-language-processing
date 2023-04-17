import os
import nltk
import nltk.corpus

print(os.listdir(nltk.data.find("corpora")))

AI="""
    Artificial intelligence (AI) is the ability of a computer or a robot
     controlled by a computer to do tasks that are usually done by humans
      because they require human intelligence and discernment.
    """

# >>>>>>>>>>>>>>>>>>>>>>Tokenizing<<<<<<<<<<<<<<<<<<<<<<<<

from nltk.tokenize import word_tokenize

AI_TOKENS=word_tokenize(AI)
print(AI_TOKENS)

from nltk.probability import FreqDist
fdist=FreqDist()

for word in AI_TOKENS:
    fdist[word.lower()]+=1
print(fdist)
fdist_top10=fdist.most_common()
print(fdist_top10)

from nltk.tokenize import blankline_tokenize
AI_Blank=blankline_tokenize(AI)
print(len(AI_Blank))

# *******************************Tokenization**********************************
print('*******************************Tokenization**********************************')

from nltk.util import bigrams, trigrams, ngrams
string="""
Artificial intelligence is widely used to provide personalised
 recommendations to people, based for example on their 
previous searches and purchases or other online behaviour."""

quotes_tokens=nltk.word_tokenize(string)
print(quotes_tokens)

quotes_bigrams=list(nltk.bigrams(quotes_tokens))
print(quotes_bigrams)

quotes_trigrams=list(nltk.trigrams(quotes_tokens))
print(quotes_trigrams)

quotes_ngrams=list(nltk.ngrams(quotes_tokens,4))
print(quotes_ngrams)

# >>>>>>>>>>>>>>>>>>>>>>Stemming<<<<<<<<<<<<<<<<<<<<<<<<
print('>>>>>>>>>>>>>>>>>>>>>>Stemming<<<<<<<<<<<<<<<<<<<<<<<<')

from nltk.stem import PorterStemmer
pst=PorterStemmer()

print(pst.stem('having'))

wordsToStem=['give','giving','given','gave']
for i in wordsToStem:
    print(i+":"+pst.stem(i))

from nltk.stem import LancasterStemmer

lst=LancasterStemmer()
for i in wordsToStem:
    print(i+":"+lst.stem(i))

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Lemmetization<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Lemmetization<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
# Groups together different inflected forms of a word, called lemma

from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
wordLem=WordNetLemmatizer()

print(wordLem.lemmatize('corpora'))

for i in wordsToStem:
    print(i+":"+wordLem.lemmatize(i))

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Stop Words<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Stop Words<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

from nltk.corpus import stopwords
print(stopwords.words('english'))

# Removing stopwords
print(fdist_top10)

import re
punctuation=re.compile(r'[-.?!,:;()|0-9]')

post_punctuation=[]
for words in AI_TOKENS:
    word=punctuation.sub("",words)
    if len(word)>0:
        post_punctuation.append(word)

print(len(AI_TOKENS),AI_TOKENS)
print(len(post_punctuation),post_punctuation)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>POS: Parts of speech<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
sent="Shivam is a natural when it comes to drawing"
sent_tokens=word_tokenize(sent)
for token in sent_tokens:
    print(nltk.pos_tag([token]))

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Named Entity Recoginiton<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

from nltk import ne_chunk
NE_Sent="The US president stays in the WHITE HOUSE"

NE_tokens=word_tokenize(NE_Sent)
NE_tags=nltk.pos_tag(NE_tokens)

NE_Ner=ne_chunk(NE_tags)
print(NE_Ner)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Chunking<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Chunking<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
#Picking up individual pieces of information and grouping them into bigger pieces

new="The big cat ate the little mouse who was after fresh cheese"
new_tokens=nltk.pos_tag(word_tokenize(new))
print('new_tokens',new_tokens)

grammer_np=r"NP: {<DT>?<JJ>*<NN>}"
chunk_parser=nltk.RegexpParser(grammer_np)
chunk_result=chunk_parser.parse(new_tokens)
print(chunk_result)