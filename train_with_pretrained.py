import siamese, tensorflow as tf, numpy as np, operator, pandas as pd
import argparse, json, helpers, os
from gensim.models.word2vec import Word2Vec
from tensorflow.contrib.rnn import LSTMCell, GRUCell

parser = argparse.ArgumentParser()
helpers.add_bool_args(parser, ['from_json', 'bidirectional', 'sampling'], [False, False, True])
parser.add_argument('-e', '--embeddings', required=True, help='file with word embeddings')
parser.add_argument('-m', '--model_name', required=True, type=str)
parser.add_argument('-b', '--batch', default=128, type=int)
parser.add_argument('-u', '--hidden units', default=500, type=int)
parser.add_argument('-i', '--iters', default=10000, type=int)
parser.add_argument('-d', '--display', default=100, type=int)
parser.add_argument('-f', '--file', required=True, type=str, help='file with training material')
parser.add_argument('-c', '--cell', default='lstm', type=str, choices=['lstm', 'gru'])
parser.add_argument('-v', '--cv_ratio', default=0.5, type=float)

FLAGS, _ = parser.parse_known_args()

if FLAGS.from_json:
    with open(os.path.join(FLAGS.model_name, 'config.json'), 'r') as f:
        config = json.load(f)
        
else:
    del FLAGS.from_json
    config = vars(FLAGS)
    if not os.path.isdir(FLAGS.model_name):
        os.mkdir(FLAGS.model_name)
    with open(os.path.join(FLAGS.model_name, 'config.json'), 'w') as f:
        json.dump(config, f)
        
data = pd.read_csv(config['file'])[['question1', 'question2', 'is_duplicate']].astype(str)
print('data loaded')

embedding_base = '/home/lukasz/Dokumenty/DEEP LEARNING/NLP/embeddings'
embeddings = Word2Vec.load_word2vec_format(os.path.join(embedding_base, config['embeddings']))

cell = LSTMCell if config['cell'] == 'lstm' else GRUCell

word2ind = dict(zip(embeddings.index2word, range(embeddings.syn0.shape[0])))
word2ind['<UNK>'] = embeddings.syn0.shape[0]

q1, q2 = helpers.tokenize_with_dict(data, word2ind)

model = siamese.siamese(embedding_matrix=np.vstack([embeddings.syn0, np.ones(embeddings.syn0.shape[1])]),
                        hidden_units=config['hidden units'], bidirectional=config['bidirectional'], cell=cell,
                        clipping='none')

print('model built')
del embeddings 
saver = tf.train.Saver()
N = data.shape[0]

inds = np.random.permutation(N)
split = int(N*config['cv_ratio'])

train = helpers.pair_iterator(q1[inds[:split]], q2[inds[:split]], data.ix[inds[:split], 'is_duplicate'].astype(int), batch=config['batch'])
test = helpers.pair_iterator(q1[inds[split:]], q2[inds[split:]], data.ix[inds[split:], 'is_duplicate'].astype(int), batch=config['batch']*4)

print('iterators created')

losses_test = []
accuracy_test = []
gradients_avg_norm_test = []

losses_train = []
accuracy_train = []
gradients_avg_norm_train = []


try:
    
    for i in range(config['iters']):
        
        q, a, l = train.deal()
        
        model.train(q, a, l)
        
        predictions, loss = model.infer_metrics(q, a, l)
        
        accuracy_train.append(float(np.mean(predictions==l)))
        losses_train.append(float(loss))
        gradients_avg_norm_train.append(float(np.mean([np.sum(x*x)/np.prod(x.shape) for x in model.calc_gradients(q, a, l)])))
        
        q, a, l = test.deal()
        
        predictions, loss = model.infer_metrics(q, a, l)
        
        accuracy_test.append(float(np.mean(predictions==l)))
        losses_test.append(float(loss))
        gradients_avg_norm_test.append(float(np.mean([np.sum(x*x)/np.prod(x.shape) for x in model.calc_gradients(q, a, l)])))

        if i%config['display']==0:

            print(i)
            print('test accuracy: {:.2f} gradient avg: {:.2f} loss {:.2f}'.format(accuracy_test[-1], gradients_avg_norm_test[-1], losses_test[-1] ))
            print('train accuracy: {:.2f} gradient avg: {:.2f} loss {:.2f}'.format(accuracy_train[-1], gradients_avg_norm_train[-1], losses_train[-1] ))
            print('\n')

            saver.save(model.sess, os.path.join(config['model_name'], config['model_name']), global_step=i)
            
except KeyboardInterrupt:
    print('training interrupted')
    
with open(os.path.join(config['model_name'], 'metrics.json'), 'w') as f:
    json.dump({'test': {'losses':losses_test, 'accuracy':accuracy_test, 'gradients': gradients_avg_norm_test}, 
               'train': {'losses':losses_train, 'accuracy':accuracy_train, 'gradients':
                        gradients_avg_norm_train}}, f)