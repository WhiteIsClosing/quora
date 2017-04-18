import siamese, tensorflow as tf, numpy as np, operator, pandas as pd
import argparse, json, helpers, os
from tensorflow.contrib.rnn import LSTMCell, GRUCell

parser = argparse.ArgumentParser()
helpers.add_bool_args(parser, ['from_json', 'bidirectional', 'sampling'], [False, False, True])
parser.add_argument('-hu' , '--hidden', default=100, type=int)
parser.add_argument('-e', '--embed', default=200, type=int)
parser.add_argument('-m', '--model_name', required=True, type=str)
parser.add_argument('-b', '--batch', default=128, type=int)
parser.add_argument('-i', '--iters', default=10000, type=int)
parser.add_argument('-d', '--display', default=100, type=int)
parser.add_argument('-f', '--file', required=True, type=str)
parser.add_argument('-c', '--cell', default='lstm', type=str, choices=['lstm', 'gru'])
parser.add_argument('-v', '--cv_ratio', default=0.5, type=float)
parser.add_argument('-ct', '--cutoff_type', default='count', type=str, choices=['count', 'number'])
parser.add_argument('-cc', '--cutoff_count', default=5, type=int)
parser.add_argument('-cn', '--cutoff_nr', default=int(1e4), type=int)
parser.add_argument('-in', '--inaccuracy', default=0, type=float)

FLAGS, unparsed = parser.parse_known_args()

if FLAGS.from_json:
    with open(os.path.join(FLAGS.model_name,  'config.json'), 'r') as f:
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

cell = LSTMCell if config['cell'] == 'lstm' else GRUCell
N = data.shape[0]
inds = np.random.permutation(N)

if config['cutoff_type'] == 'count':
    q1, q2, vocab_size, words_inds = helpers.tokenize(data, cutoff_count=config['cutoff_count'])
else:
    q1, q2, vocab_size, words_inds = helpers.tokenize(data, cutoff_number=config['cutoff_nr'])

split = int(N*config['cv_ratio'])

with open(os.path.join(config['model_name'], 'words.json'), 'w') as f:
    json.dump(words_inds, f)

train = helpers.pair_iterator(q1[inds[:split]], q2[inds[:split]], data.ix[inds[:split], 'is_duplicate'].astype(int), batch=config['batch'])
test = helpers.pair_iterator(q1[inds[split:]], q2[inds[split:]], data.ix[inds[split:], 'is_duplicate'].astype(int), batch=config['batch']*4)
print('iterators created')

model = siamese.siamese(hidden_units=config['hidden'], embedding_size=config['embed'], vocab_size=vocab_size, cell=cell, 
                bidirectional=config['bidirectional'], clipping='none')

saver = tf.train.Saver()

losses_test = []
probs_correct_test = []
accuracy_test = []
gradients_avg_norm_test = []

losses_train = []
probs_correct_train = []
accuracy_train = []
gradients_avg_norm_train = []

print([x.name for x in tf.global_variables()])

try:
    for i in range(config['iters']):

        q, a, l = train.deal()

        model.train(q, a, l)
        
        losses_train.append(float(model.infer_loss(q, a, l)))
        probs_correct_train.append(float(np.mean(model.infer_probs(q, a, l))))
        accuracy_train.append(float(np.mean(model.infer_accuracy(q, a, l))))
        gradients_avg_norm_train.append(float(np.mean([np.sum(x*x)/np.prod(x.shape) for x in model.calc_gradients(q, a, l)])))

        q, a, l = test.deal()

        losses_test.append(float(model.infer_loss(q, a, l)))
        probs_correct_test.append(float(np.mean(model.infer_probs(q, a, l))))
        accuracy_test.append(float(np.mean(model.infer_accuracy(q, a, l))))
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
    json.dump({'test': {'losses':losses_test, 'probs_correct':probs_correct_test, 'accuracy':accuracy_test, 'gradients': gradients_avg_norm_test}, 
               'train': {'losses':losses_train, 'probs_correct':probs_correct_train, 'accuracy':accuracy_train, 'gradients':
                        gradients_avg_norm_train}}, f)
