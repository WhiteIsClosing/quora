import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import GRUCell, LSTMCell, LSTMStateTuple

class siamese:
    
    def __init__(self, hidden_units, embedding_size=None, vocab_size=None, lr=1e-1, clipping='norm', clip_val = 1,
                cell=LSTMCell, bidirectional=False, trainable_embed=False, embedding_matrix=None):
        
        assert clipping in ['none', 'value', 'norm']
        
        if clipping == 'value':
            assert isinstance(clip_val, list) and len(clip_val) == 2
        if clipping == 'norm':
            assert (isinstance(clip_val, int) or isinstance(clip_val, float) ) and clip_val > 0
            
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        
        self.vocab_size = vocab_size
        
        if trainable_embed:
            self.embedding = tf.Variable(tf.truncated_normal([self.vocab_size, embedding_size]), name='embedding')
        else:
            self.embedding = tf.Variable(embedding_matrix, trainable=False, dtype=tf.float32)
            
        self.PAD = self.embedding.get_shape()[0].value-1 #last value is prepared just for padding
            
        self.question_ph = tf.placeholder(tf.int32, [None, None])
        self.answer_ph = tf.placeholder(tf.int32, [None, None])
        
        self.question = tf.nn.embedding_lookup(self.embedding, self.question_ph)
        self.answer = tf.nn.embedding_lookup(self.embedding, self.answer_ph)
        
        self.targets_ph = tf.placeholder(tf.int32, [None])
        
        self.lengths_q = tf.placeholder(tf.int32, [None])
        self.lengths_a = tf.placeholder(tf.int32, [None])
      
        if bidirectional:
            self.cell_fw = cell(num_units=hidden_units)
            self.cell_bw = cell(num_units=hidden_units)
            
            with tf.variable_scope('twins') as scope:
                
                self.outs_q, self.states_q = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.cell_fw,
                    cell_bw=self.cell_bw,
                    sequence_length=self.lengths_q,
                    inputs=self.question,
                    dtype=tf.float32)
                
                scope.reuse_variables()
                
                self.outs_a, self.states_a = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.cell_fw,
                    cell_bw=self.cell_bw,
                    sequence_length=self.lengths_q,
                    inputs=self.question,
                    dtype=tf.float32)
                
                
                if isinstance(self.states_q[0], LSTMStateTuple):
                    
                    sq_fw, sq_bw = self.states_q
                    sa_fw, sa_bw = self.states_a
                    
                    sqc = tf.concat([sq_fw.c, sq_bw.c], axis=1)
                    sac = tf.concat([sa_fw.c, sa_bw.c], axis=1)
                    sqh = tf.concat([sq_fw.h, sq_bw.h], axis=1)
                    sah = tf.concat([sa_fw.h, sa_bw.h], axis=1)
                    
                    self.states_q = LSTMStateTuple(c=sqc, h=sqh)
                    self.states_a = LSTMStateTuple(c=sac, h=sqh)
                    
                    
                else:
                    
                    self.states_q = tf.concat(self.states_q, axis=1)
                    self.states_a = tf.concat(self.states_a, axis=1)
            
        else:
            
            self.siamese_cell = cell(num_units=hidden_units)
        
            with tf.variable_scope('twins') as scope:
                self.outs_q, self.states_q = tf.nn.dynamic_rnn(
                    cell=self.siamese_cell,
                    sequence_length=self.lengths_q,
                    inputs=self.question,
                    dtype=tf.float32)

                scope.reuse_variables()

                self.outs_a, self.states_a = tf.nn.dynamic_rnn(
                    cell=self.siamese_cell,
                    sequence_length=self.lengths_a,
                    inputs=self.answer,
                    dtype=tf.float32)

                
                
        if isinstance(self.states_q, LSTMStateTuple):
            
            self.states_q = tf.concat([self.states_q.c, self.states_q.h], axis=1)
            self.states_a = tf.concat([self.states_a.c, self.states_a.h], axis=1)

        state_size = self.states_q.get_shape()[1].value
            
        self.distance_weights = tf.Variable(tf.truncated_normal([state_size, 2], stddev=1), name='dist_W')
        self.distance_bias = tf.Variable(tf.ones([2]), name='dist_B')
        
        self.intermediate = tf.matmul(tf.abs(self.states_q - self.states_a), self.distance_weights) + self.distance_bias
        
#         self.distance = tf.nn.sigmoid( self.intermediate )
        
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.intermediate, labels=self.targets_ph)
        
        self.opt = tf.train.AdamOptimizer(lr)
        
        self.grads_vars = self.opt.compute_gradients(tf.reduce_mean(self.loss))
        if clipping == 'none':
            self.optimize = self.opt.minimize(tf.reduce_mean(self.loss))
            
        elif clipping in ['value', 'norm']:
            
            if clipping == 'value':
                self.clipped_grads = [ (tf.clip_by_value(grad, clip_val[0], clip_val[1]), var) for grad, var in self.grads_vars]
            
            else:
                self.clipped_grads = [ (tf.clip_by_norm(grad, clip_by_val), var) for grad, var in self.grads_vars]
 
            self.optimize = self.opt.apply_gradients(self.clipped_grads)
            
#         self.gradients_norms = [tf.nn.l2_loss(x[0]) for x in self.grads_vars]
        
        self.sess.run(tf.global_variables_initializer())
        
    def transform_inputs(self, x, y, targets):
        
#         targets = np.expand_dims(targets, axis=1)
        
        import copy
        X = copy.deepcopy(x)
        Y = copy.deepcopy(y)
        
        lx = list(map(len, X))
        ly = list(map(len, Y))
        
        mx = max(lx)
        my = max(ly)
        
        N = len(X)
        
        temp_x = np.zeros((N, mx), dtype=np.int32)
        temp_y = np.zeros((N, my), dtype=np.int32)
        
        if isinstance(X[0], list):
            for i in range(N):
                
                X[i] = X[i] + [self.PAD]*(mx-lx[i])
                Y[i] = Y[i] + [self.PAD]*(my-ly[i])
                
                temp_x[i, :] = np.asarray(X[i], dtype=np.int32)
                temp_y[i, :] = np.asarray(Y[i], dtype=np.int32)
                
        else:
            for i in range(N):

                X[i] = X[i].tolist() + [self.PAD]*(mx-lx[i])
                Y[i] = Y[i].tolist() + [self.PAD]*(my-ly[i])
                
                temp_x[i, :] = np.asarray(X[i], dtype=np.int32)
                temp_y[i, :] = np.asarray(Y[i], dtype=np.int32)
            
        
        return {self.question_ph:temp_x, self.answer_ph:temp_y, self.lengths_q:lx, self.lengths_a:ly, self.targets_ph:targets}
        
    def train(self, questions, answers, targets):
        
        fd = self.transform_inputs(questions, answers, targets)
        
        self.sess.run(self.optimize, feed_dict=fd)
        
    def run_op(self, op, fd):
        
        return self.sess.run(op, feed_dict=fd)
    
    def reset(self):
        
        self.sess.run(tf.global_variables_initializer())
        
    def infer_class(self, questions, answers):
        
        fd = self.transform_inputs(questions, answers)
        
        return self.sess.run(tf.argmax(self.intermediate,1), feed_dict=fd)
        
    def infer_loss(self, questions, answers, targets, **kwargs):
        
        fd = self.transform_inputs(questions, answers, targets)
        
        return self.sess.run(tf.reduce_mean(self.loss), feed_dict=fd)
    
    def infer_accuracy(self, questions, answers, targets):
        
        fd = self.transform_inputs(questions, answers, targets)
        
        return self.sess.run(tf.argmax(self.intermediate, 1), feed_dict=fd)==targets
    
    def infer_metrics(self, questions, answers, targets):
        
        fd = self.transform_inputs(questions, answers, targets)
        
        return self.sess.run([tf.argmax(self.intermediate,1), 
                              tf.reduce_mean(self.loss)], feed_dict=fd)
    
    def infer_probs(self, questions, answers, targets):

        N = len(questions)
        fd = self.transform_inputs(questions, answers, targets)
        probs = self.sess.run(tf.nn.softmax(self.intermediate), feed_dict=fd)
        return probs[range(N), targets]
    
    def calc_gradients(self, questions,answers, targets):
        
        fd = self.transform_inputs(questions, answers, targets)
        
        pre_g = self.sess.run([x[0] for x in self.grads_vars], feed_dict=fd)
        N = len(pre_g)
        for i in range(N):
            if not isinstance(pre_g[i], np.ndarray):
                #counter-effort against IndexedSlicesValues, appearing alongside nn.embedding_lookup
                pre_g[i] = pre_g[i][0]
        return pre_g

    def predict(self, questions, answers, **kwargs):
        
        questions = [u[:self.max_qL] for u in questions]
        answers = [u[:self.max_aL] for u in answers]
        
        lengths_q = list(map(len, questions))
        lengths_a = list(map(len, answers))
        
        return self.sess.run(self.distance, feed_dict={self.question_ph:questions, self.answer_ph:answers, 
                            self.lengths_q:lengths_q, self.lengths_a:lengths_a}, **kwargs)

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    model = siamese(54, 12, 43, clipping='none')
    
    losses = []
    grads = []

    def generate_fake_data():

        B = 128
        labels = [0]*B + [1]*B

        L = np.random.randint(3,12, B).tolist()

        xs = []
        ys = []
        for i in range(B):

            xs.append(np.random.randint(1, model.vocab_size, L[i]))
            ys.append(xs[-1])

        for i in range(B):

            xs.append(np.random.randint(1, model.vocab_size, L[i]))
            ys.append(xs[-1][::-1])

        return xs, ys, labels

    for i in range(1000):

        xs, ys, labels = generate_fake_data()

        model.train(xs, ys, labels)
        
        losses.append(model.infer_loss(xs, ys, labels))
        grads.append(model.calc_gradients(xs, ys, labels))
        
        if i%100==0:
            print(np.mean(losses[-100:]))
        
        
    plt.plot(range(100), losses)
    plt.show()