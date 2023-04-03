import ssl
import nltk
import urllib.request

# ignore SSL certificate errors
ssl._create_default_https_context = ssl._create_unverified_context

# set the proxy
proxy = urllib.request.ProxyHandler({'http': 'http://165.225.112.14:10015', 'https': 'http://165.225.112.14:10015'})
opener = urllib.request.build_opener(proxy)
urllib.request.install_opener(opener)

nltk.download('punkt')
nltk.download('wordnet')