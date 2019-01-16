from grammar import *


words = [ make_embedded_reber() for i in range(6000) ]
max_len = len(max(words,key=len))
sl, words_one_hot, next_chars_one_hot = zip(*[ \
        (len(i) ,\
        np.pad(str_to_vec(i),((0,max_len-len(i)),(0,0)),mode='constant'), \
        np.pad(str_to_next_embed(i),((0,max_len-len(i)),(0,0)),mode='constant')) \
        for i in words ])

print(words_one_hot)
