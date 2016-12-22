from nltk.corpus import wordnet

def load_word_vectors(file_name):
    print "Loading vectors..."
    f = open(file_name,'r')
    words = []
    word_map = {}
    i=0
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        words.append(word)
        word_map[word]=i
        i=i+1
    print "Done.", len(word_map)," words loaded!"
    return word_map, words

print 'Getting synonyms...'
input_file_name = sys.argv[1]
output_file_name = sys.argv[2]
word_map, words = load_word_vectors()
word_syn = dict()
for w in words:
    syn_set = set()
    for syn in wordnet.synsets(w):
        for l in syn.lemmas:
            syn_set.append(l.name())
    word_syn[w]=syn_set
# Save codes in a file
print 'Writing in a file...'
i=0
f=open(output_file_name, 'w')
for w in vocab_orig:
    vec_str=''
    for s in word_syn[w]:
        vec_str= " ".join(s)
    f.write(str(word_map[w])+' '+w+' '+vec_str+'\n')
    if i%10000==0:
        print 'i =',i
    i=i+1
