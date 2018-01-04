import re
import load

from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer

url_tag = "__url__"
path_tag = "__path__"

tweet_tokenize = TweetTokenizer(False)
lemmatizer = WordNetLemmatizer()

path_regex = re.compile("(?:\s|[\"'(,])(?:~|[.]{1,2}|)?(?:\/[^ \"'!?;,*:<>|]+){2,}/?")
strict_html = re.compile("https?://(?:[a-zA-Z0-9\-]*[.])?([a-zA-Z0-9\-]*)(?:[.](?:[a-z]{2,13}))+(?:/(?:[a-zA-Z0-9$%*+\:\;\=\?\^_@.&+~#\-]|(?:%[0-9a-fA-F][0-9a-fA-F]))*)*/?")


def add_tag(sentence, domain=True):
    res = sentence
    if domain:
        res = strict_html.sub(lambda m: "{} {} __{}__ ".format(m.group(0), url_tag, m.group(1)), res)
    else:
        res = strict_html.sub(lambda m: "{} {} ".format(m.group(0), url_tag), res)

    res = path_regex.sub(lambda m:"{} {} ".format(m.group(0), path_tag), res)
    return res


def tokenize_all(sentences, verbose_at=100, add_tag = False, domain_tag = False):
    tokenized = []
    i = 0
    for sent in sentences:
        tokenized.append(tokenize(sent, add_tag, domain_tag))
        i += 1
        if i % verbose_at == 0:
            print("Processed {}".format(i), end="\r")
    
    print()
    return tokenized

def tokenize(sentence, do_add_tag = False, domain_tag = False):
    res = add_tag(sentence, domain_tag) if do_add_tag else sentence
    res = tweet_tokenize.tokenize(res)
    return [lemmatizer.lemmatize(word) for word in res]


if __name__ == "__main__":
    # TEST: show mappings of url and server name in file data/urls.txt
    data = load.load_csv("data/ubuntu_train_1.csv")

    res = tokenize_all(data[:,0], verbose_at=1000, add_tag=False)
