#let us try to open the text document for analysing word frequency
document1 = open("empmail.txt","r")

# Read the document and print its contents
document_text = document1.read()
#print(document_text)

#For word frequency analysis, we need to normalize the text by removing all numeric digits and punctuations
from string import punctuation

# remove numeric digits
result = ''.join(character for character in document_text if not character.isdigit())

# remove punctuation and make lower case
result = ''.join(character for character in result if character not in punctuation).lower()

# print the normalized text
print(result)

#Its time to get the word frequency. 
#So we need to split this email into induvidual words and get the frequency analysed
#for the same, we rely on two modules: which are nltk(natural language toolkit) and pandas
import nltk
import pandas as friendly_panda
from nltk.probability import FreqDist
nltk.download("punkt")
#Now, you may be thinking what is this 'Punkt'? It is a pre-trained tokenizer for English launguage

#we see that there are many stopping words, such as and and the. So, let us go ahead and remove such stop words.
nltk.download("stopwords")
from nltk.corpus import stopwords
result = ' '.join([word for word in result.split() if word not in (stopwords.words('english'))])
print("\n")
print(result)

#Ok, back to business. Tokenize the text into individual words
words = nltk.tokenize.word_tokenize(result)

#after this we will get the frequency distribution of words, into a dataframe
frequency_distribution = FreqDist(words)

#For people new to Pandas, this is how a pandas DataFrame command looks like:
#class pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
#An example is: "Square DataFrame with homogeneous dtype"
# d1 = {'col1': [1, 2], 'col2': [3, 4]}
# df1 = pd.DataFrame(data=d1)
# df1
#so your results are:
#   col1  col2
#0     1     3
#1     2     4

#if you use T to transpose index, then the result look like this;
#      0  1
#col1  1  2
#col2  3  4

#Coming back to our code. Now, we need to count the no. in dataframe
countf = friendly_panda.DataFrame(frequency_distribution, index =[0]).T

#You may be wondering now, as what does 'T' do here. It will Transpose Index and columns
countf.columns = ['Count']
print (countf)
 
