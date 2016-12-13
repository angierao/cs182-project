import numpy as np
import random
import time
import sys
import tensorflow as tf

# Helper Functions
def embedding(data_, vocab):
	data = np.zeros((len(data_), len(vocab)))
	cnt = 0
	for s in data_:
		v = [0.0] * len(vocab)
		v[vocab.index(s)] = 1.0
		data[cnt, :] = v
		cnt += 1
	return data

def decode_embed(array, vocab):
	return vocab[array.index(1)]

# LSTM RNN Network
class LSTM_RNN:
	def __init__(self, 
			     input_size, 
			     lstm_size, 
			     num_layers, 
			     output_size, 
			     session, 
			     learning_rate = 0.003, 
			     name="rnn"):
		# initialize values
		self.input_size = input_size
		self.lstm_size = lstm_size
		self.num_layers = num_layers
		self.output_size = output_size
		self.session = session
		self.learning_rate = tf.constant(learning_rate)
		self.scope = name

		self.last_state = np.zeros((self.num_layers * 2 * self.lstm_size,))
		with tf.variable_scope(self.scope):
			self.x_input = tf.placeholder(tf.float32, 
									      shape=(None, None, self.input_size), 
									      name="x_input")
			self.init_value = tf.placeholder(tf.float32, 
				                             shape=(None, self.num_layers * 2 * self.lstm_size),
				                             name="init_value")
			
			self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size, 
														  forget_bias=1.0, 
														  state_is_tuple=False)
			self.lstm = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell] * self.num_layers, 
													 state_is_tuple=False)
			outputs, self.lstm_new_state = tf.nn.dynamic_rnn(self.lstm, 
															 self.x_input, 
															 initial_state=self.init_value)
			self.rnn_out_W = tf.Variable(tf.random_normal((self.lstm_size, self.output_size), stddev=0.01 ))
			self.rnn_out_B = tf.Variable(tf.random_normal((self.output_size,), stddev=0.01))
			outputs_reshaped = tf.reshape(outputs, [-1, self.lstm_size])
			network_output = (tf.matmul(outputs_reshaped, self.rnn_out_W) + self.rnn_out_B)
			batch_time_shape = tf.shape(outputs)
			self.final_outputs = tf.reshape( tf.nn.softmax( network_output), (batch_time_shape[0], batch_time_shape[1], self.output_size) )
			
			# training
			self.y_batch = tf.placeholder(tf.float32, (None, None, self.output_size))
			y_batch_long = tf.reshape(self.y_batch, [-1, self.output_size])
			self.loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(network_output, y_batch_long) )
			self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, 0.9).minimize(self.loss)


	def run_step(self, x, init_zero_state=True):
		if init_zero_state:
			init_value = np.zeros((self.num_layers * 2 * self.lstm_size,))
		else:
			init_value = self.last_state
		out, next_lstm_state = self.session.run([self.final_outputs, self.lstm_new_state], feed_dict={self.x_input:[x], self.init_value:[init_value]   } )
		self.last_state = next_lstm_state[0]
		return out[0][0]

	def train_batch(self, xbatch, ybatch):
		init_value = np.zeros((xbatch.shape[0], self.num_layers*2*self.lstm_size))
		loss, _ = self.session.run([self.loss, self.optimizer], feed_dict={self.x_input:xbatch, self.y_batch:ybatch, self.init_value:init_value   } )
		return loss


# Generating Text
def gen_text(my_prefix = "The "):
	my_prefix = my_prefix.lower()
	for i in range(len(my_prefix)):
		out = net.run_step(embedding(my_prefix[i], vocab) , i == 0)

	print "TEXT:"
	ans = my_prefix
	for i in range(LEN_TEST_TEXT):
		element = np.random.choice(range(len(vocab)), p=out) 
		ans += vocab[element]
		out = net.run_step(embedding(vocab[element], vocab) , False)
	print ans


OK = True
data_ = ""
with open('../../data/trump_sentences.txt', 'r') as f:
	data_ += f.read()
data_ = data_.lower()

vocab = list(set(data_))
data = embedding(data_, vocab)
input_size = len(vocab)
output_size = len(vocab)
lstm_size, num_layers, batch_size, time_steps = 256, 2, 64, 100
RUN_BATCHES = 10000
LEN_TEST_TEXT = 500

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
tf_session = tf.InteractiveSession(config=config)

net = LSTM_RNN(input_size = input_size,
					lstm_size = lstm_size,
					num_layers = num_layers,
					output_size = output_size,
					session = tf_session,
					learning_rate = 0.003,
					name = "char_rnn_network")

tf_session.run(tf.initialize_all_variables())

saver = tf.train.Saver(tf.all_variables())


if OK:
	last_time = time.time()
	batch = np.zeros((batch_size, time_steps, input_size))
	batch_y = np.zeros((batch_size, time_steps, input_size))

	possible_batch_ids = range(data.shape[0]-time_steps-1)
	for i in range(RUN_BATCHES):
		batch_id = random.sample(possible_batch_ids, batch_size)

		for j in range(time_steps):
			ind1 = [k+j for k in batch_id]
			ind2 = [k+j+1 for k in batch_id]
			batch[:, j, :] = data[ind1, :]
			batch_y[:, j, :] = data[ind2, :]

		cst = net.train_batch(batch, batch_y)

		if (i % 100) == 0:
			new_time = time.time()
			diff = new_time - last_time
			last_time = new_time

			print "RUNNING BATCH #",i,"   loss: ",cst," at ",(100.0/diff)," batches / s"
			saver.save(tf_session, "../../data/model.ckpt")
			gen_text()


