import random, numpy as np, operator

def add_bool_args(parser, things, defaults):
    
    if isinstance(things, str):
        things = [things]
       
    if isinstance(defaults, bool):
        defaults = [defaults]
        
    for th in things:
        
        parser.add_argument('--' + th, dest=th, action='store_true')
        parser.add_argument('--no_' + th, dest=th, action='store_false')
        
    parser.set_defaults(**dict(zip(things, defaults)))
    
class pair_iterator:
    
    def __init__(self, x, y, labels, random=False, batch=32, inaccuracy=0):
        
        assert len(x) == len(y)
        
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.labels = np.asarray(labels)
        
        
        N = len(x)
        
        if random:
            indices = np.random.permutation(N)
            self.x, self.y, self.labels = self.x[indices], self.y[indices], self.labels[indices]
            
        self.random = random
        self.batch = batch
        self.N = N
        
    def deal(self):
        
        inds = random.sample(range(self.N), self.batch)
        
        return self.x[inds], self.y[inds], self.labels[inds]
        

def tokenize(df, cutoff_count=None, cutoff_number=None):
    
    assert (cutoff_count is not None)^(cutoff_number is not None)
    
    word_count = {}

    for col in ['question1', 'question2']:

        for q in df[col].astype(str):

            seq = q.split()

            for w in seq:
                if w not in word_count:
                    word_count[w] = 0

                word_count[w] += 1
                
    words_with_counts = sorted(word_count.items(), key=operator.itemgetter(1))[::-1]
    vocab, words_nrs = list(zip(*words_with_counts))
    
    if cutoff_count is not None:
        cutoff_number = words_nrs.index(cutoff_count)
    
    vocab = vocab[:cutoff_number]
    
    words_inds = dict(zip(['<UNK>'] + list(vocab), list(range(1, len(vocab)+2))))
    
    final = {'question1':[], 'question2':[]}
    
    for col in ['question1', 'question2']:
        
        for q in df[col]:
            
            seq = q.split()
            
            final[col].append([words_inds[x] if x in words_inds else words_inds['<UNK>'] for x in seq])
            
    return np.asarray(final['question1']), np.asarray(final['question2']), len(words_inds)+1, words_inds

def tokenize_with_dict(df, dictionary, with_unk=False):
    
    final = {'question1':[], 'question2':[]}
    
    for col in ['question1', 'question2']:
        
        for q in df[col]:
            
            seq = q.split()
            
            if with_unk:
                final[col].append([dictionary[x] if x in dictionary else dictionary['<UNK>'] for x in seq])
            else:
                final[col].append([dictionary[x] for x in seq if x in dictionary])
            
    return np.asarray(final['question1']), np.asarray(final['question2'])
