#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string
from nltk.stem.snowball import SnowballStemmer



try:
    maketrans = ''.maketrans
except AttributeError:
    # fallback for Python 2
    from string import maketrans

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = " "
    
    stemmer = SnowballStemmer("english")
    
    
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(maketrans("", "",string.punctuation))
        text_string = str.split(text_string)
        ### project part 2: comment out the line below
        # words = text_string
        for str_ in text_string:            
                stem_str = stemmer.stem(str_)                                                
                words += stem_str + " "
        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        
    



    return words

    

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print(text)



if __name__ == '__main__':
    main()

