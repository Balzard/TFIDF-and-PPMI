# -*- coding: utf-8 -*-
from nltk.corpus.reader import PlaintextCorpusReader
from nltk.lm import Vocabulary
import math
import copy
from gensim.models import Word2Vec
from numpy import dot
from numpy.linalg import norm

# Ok
def get_context(text: list, target: int, size_context: int) -> list:
  """Create context window

  Args:
      text (list): text corpus as single sequence of words
      target (int): position in text
      size_context (int): number of word a the right or left of the target word in the contextual window

  Returns:
      list: context window
  """
  
  text_size = len(text)
  
  if target < size_context:
      context = [text[i] for i in range(target)] + [text[i] for i in range(target + 1, target + size_context + 1)]
      
  elif target > text_size - 1 - size_context:
    context = [text[i] for i in range(target - size_context, target)] + [text[i] for i in range(target + 1, len(text))]
    
  else:
    context = [text[i] for i in range(target - size_context, target)] + [text[i] for i in range(target + 1, target + size_context + 1)]
    
  return context


# Ok
def get_cooccurence_matrix(text: list, window_size: int) -> dict:
  """Return co-occurence matrix from a text corpus without storing zero entries

  Args:
      text (list): text corpus as a single sequence of words
      window_size (int): size of contextual windows (assumed to be odd)

  Returns:
      dict: coocurrence matrix where each key is a target word that contains its context words with their count
  """

  size_context = round(window_size/2) # number of element to a side of the target word in the context window
  voc = Vocabulary(text)
  words = sorted(voc.counts) # no UNK token
  matrix = {k:{} for k in words}

  for target in range(len(text)):
    
    context = get_context(text, target, size_context)
      
    for word in context:

      if matrix[text[target]] == {} or word not in matrix[text[target]].keys():
        matrix[text[target]][word] = 1

      else:
        matrix[text[target]][word] += 1

  return matrix


# seems OK too
def compute_df(text: list, words: list, windows_size: int) -> dict:
  """Compute the number of contextual windows of any target word where this context word occurs

  Args:
      text (list): text corpus as a single sequence of text
      words (list): words of text corpus
      windows_size (int): size of contextual windows (assumed to be odd)

  Returns:
      dict: dictionnary where each key is a word and its value is the number of contextual windows where this word occurs
  """
  
  df = {k:0 for k in words}
  size_context = round(windows_size/2)
  
  for target in range(len(text)):
    
    context = set(get_context(text, target, size_context))
    
    for word in context:
      df[word] += 1
      
  return df
    
    
# seems OK too
def tf_idf(matrix: dict, text: list, window_size: int) -> dict:
  """Return TF-IDF matrix based on co-occurence matrix

  Args:
      matrix (dict): co-occurence matrix
      text (list): text corpus as a single sequence of words
      words (list): words of text corpus
      window_size (int): size of contextual windows (assumed to be odd)

  Returns:
      dict: TF-IDF matrix where each key is a target word that contains its context words with their weighted frequencies
  """
  
  result = copy.deepcopy(matrix)
  N = len(text)
  df = compute_df(text, words, window_size)

  for target in result.keys():
    for word in result[target]:
      tf = math.log10(result[target][word] + 1)

      idf = math.log10(N/df[word])
      result[target][word] =  tf * idf

  return result


def ppmi(matrix: dict) -> dict:
  """[summary]

  Args:
      matrix (dict): [description]

  Returns:
      dict: [description]
  """
  
  result = copy.deepcopy(matrix)
  epsilon = 0.0001
  nb_words = len(result.keys())
  all_prob = []
  p_wi = {}
  p_cj = {}

  for target in result.keys():
    tmp = [(result[target][word] + epsilon) for word in result[target]]
    zero_entries = nb_words - len(tmp)
    tmp.append(zero_entries * epsilon)
    all_prob.append(sum(tmp))

  sum_prob = sum(all_prob) 
  zero_count_prob = epsilon / sum_prob # prob when context word never appear with target word

  for target in result.keys():
    for word in result[target]:
      p_w_c = (result[target][word] + epsilon) / sum_prob
      result[target][word] = p_w_c

  for target in result.keys():
    p_wi[target]= sum([result[target][i] for i in result[target]])

    for word in result[target]:
      p_c = sum([result[i][j] for i in result.keys() for j in result[i] if j == word and i != word])

      result[target][word] = max(math.log2(result[target][word]/(p_w*p_c)), 0)
      
  return result


def cosine_similarity(v1: list, v2: list) -> float:
  """Compute cosine similarity between two vectors

  Args:
      v1 (list or numpy array): vector 1
      v2 (list or numpy array): vector 2

  Returns:
      float: cosine similarity 
  """
  return dot(v1, v2)/(norm(v1)*norm(v2))


def to_vector(v: dict, words: list) -> list:
  """Transform an entry from co-occurence, tf-idf or ppmi matrix intro a vector

  Args:
      v (dict): entry of the matrix
      words (list): words of the corpus

  Returns:
      list: vector
  """

  v = [(words.index(k),v) for k,v in v.items()]
  v = sorted(v, key=lambda tup: tup[0])
  words_in_vector = [i[0] for i in v]
  words_not_in_vector = [(i,0) for i in range(len(words)) if i not in words_in_vector]
  result = sorted(v + words_not_in_vector, key=lambda tup: tup[0])
  result = [i[1] for i in result]
  return result


def get_top(word: str, words: list, tfidf: dict, top_nb: int) -> list:
  """Return the most similar words to word

  Args:
      word (str): word to compare
      words (list): words of the corpus
      tfidf (dict): tf_idf matrix
      top_nb (int): number of most similar words to return

  Returns:
      list: list of tuples (word, cosine_similarity)
  """

  vec_word = to_vector(tfidf[word], words)
  result = []
  for target in tfidf.keys():
    if target != word:
      vec_target = to_vector(tfidf[target], words)
      cos = cosine_similarity(vec_word, vec_target)

    if len(result) < top_nb:
      result.append((target, cos))
    else:
      if cos > min(result, key=lambda t:t[1])[1]:
        result.remove(min(result, key=lambda t:t[1]))
        result.append((target, cos))

  return result



if __name__ == "__main__":
  
  
  corpus = PlaintextCorpusReader(root="", fileids=["p2_sherlock.txt"])

  punc =  ["\"", "'", ",", ".", ":", "?", "!", ".\"", "?\"", "!\"", "-", ",\"", "--"]
  text = [word.lower() for word in corpus.words() if word not in punc]
  voc = Vocabulary(text)
  voc_size = len(voc.counts) # no UNK token
  words = sorted(voc.counts) # no UNK token
  
  
  matrix = get_cooccurence_matrix(text, 5)
  df = compute_df(text, words, 5)
  tf = tf_idf(matrix, text, 5)
  print(get_top("hundred", words, tf, 5))