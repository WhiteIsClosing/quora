import siamese, os, json, argparse, helpers, pandas as pd, tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', required=True, type=str)
parser.add_argument('-s', '--step', default=-1, type=int)
parser.add_argument('-b', '--batch', default=128, type=int)
parser.add_argument('-f', '--file', default='preprocessed_test.csv', type=str)

FLAGS, _ = parser.parse_known_args()

with open(os.path.join(FLAGS.model_name, 'config.json'), 'r') as f:
    config = json.load(f)
    
with open(os.path.join(FLAGS.model_name, 'words.json'), 'r') as f:
    words_indices = json.load(f)
    
data = pd.read_csv(FLAGS.file)
print('loaded data')

cell = LSTMCell if config['cell'] == 'lstm' else GRUCell

model = siamese.siamese(hidden_units=config['hidden'], embedding_size=config['embed'], cell=cell, 
                bidirectional=config['bidirectional'], clipping='none', vocab_size=len(words_indices)+1)

print([x.name for x in tf.global_variables()])


steps = sorted([x.lstrip(FLAGS.model_name + '-').rstrip('.index') for x in os.listdir(FLAGS.model_name) if x.startswith(FLAGS.model_name) and x.endswith('index')])


if FLAGS.step == -1:
    restore_step = steps[-1]

else:
    restore_step = FLAGS.step
    
saver = tf.train.Saver()
saver.restore(model.sess, os.path.join(FLAGS.model_name, FLAGS.model_name + '-' + str(restore_step)))

q1, q2 = helpers.tokenize_with_dict(data, words_indices)

N = data.shape[0]
offset = 0
b = config['batch']
final_df = pd.DataFrame(columns=['test_id', 'is_duplicate'])

for i in range(N//b+1):
    
    results = model.infer_class(q1[offset:offset+b], q2[offset:offset+b])
    final_df = pd.concat([final_df, pd.DataFrame({'test_id':np.arange(offset:offset+b), 'is_duplicate':results.astype(int)})], axis=0)
    
    