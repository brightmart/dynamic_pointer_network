# -*- coding: utf-8 -*-
# Pointer Network: 1.word embedding 2.encoder 3.decoder(optional with attention). for more detail, please check: Pointer Network https://arxiv.org/pdf/1506.03134.pdf
import tensorflow as tf
import numpy as np
import tensorflow.contrib as tf_contrib
import random
import copy
import os

class dynamic_pointer_net:
    def __init__(self, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                 vocab_size, embed_size,hidden_size, is_training,decoder_sent_length=6,
                 initializer=tf.random_normal_initializer(stddev=0.1),clip_gradients=5.0,l2_lambda=0.0001):
        """init all hyperparameter here"""
        # set hyperparamter
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.5)
        self.initializer = initializer
        self.decoder_sent_length=decoder_sent_length
        self.hidden_size = hidden_size
        self.clip_gradients=clip_gradients
        self.l2_lambda=l2_lambda

        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")                 #x
        self.query=tf.placeholder(tf.int32,[None,self.decoder_sent_length])                                       #query ADD
        self.decoder_input = tf.placeholder(tf.int32, [None, self.decoder_sent_length],name="decoder_input")  #y, but shift
        self.input_y_label = tf.placeholder(tf.int32, [None, self.decoder_sent_length], name="input_y_label") #y, but shift
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference() #logits shape:[batch_size,decoder_sent_length,self.sequence_length]

        self.predictions = tf.argmax(self.logits, axis=2, name="predictions")
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y_label)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
        if not is_training:
            return
        self.loss_val = self.loss_seq2seq()
        self.train_op = self.train()

    def inference(self):
        """main computation graph here:
        #1.Word embedding. 2.Encoder with GRU 3.Decoder using GRU(optional with attention)."""
        # 1.embedding of words
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)  #[None, self.sequence_length, self.embed_size]
        # 2.encoder with GRU
        # 2.1 forward gru
        hidden_state_forward_list = self.gru_forward(self.embedded_words,self.gru_cell)  # a list,length is sentence_length, each element is [batch_size,hidden_size]
        # 2.2 backward gru
        hidden_state_backward_list = self.gru_forward(self.embedded_words,self.gru_cell,reverse=True)  # a list,length is sentence_length, each element is [batch_size,hidden_size]
        # 2.3 concat forward hidden state and backward hidden state. hidden_state: a list.len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
        thought_vector_list=[tf.concat([h_forward,h_backward],axis=1) for h_forward,h_backward in zip(hidden_state_forward_list,hidden_state_backward_list)]#list,len:sent_len,e:[batch_size,hidden_size*2]
        # 3.Decoder using GRU with attention
        thought_vector_raw=tf.stack(thought_vector_list,axis=1) #shape:[batch_size,sentence_length,hidden_size*2]

        attention_states=thought_vector_raw #[None, self.sequence_length, self.embed_size]
        embedded_query = tf.nn.embedding_lookup(self.Embedding, self.query)                 #[batch_size,self.decoder_sent_length,embed_size]
        embedded_query_squeezed = self.gru_forward(embedded_query, self.gru_cell,scope="gru_forward_query") # a list of 2d, each is [batch_size, hidden_size]
        embedded_query_squeezed=[embedded_query_squeezed[-1]]*self.decoder_sent_length # use last hidden state to represent query; and replicate many copy, same length as decode length
        #embedded_query_splitted = tf.split(embedded_query, self.decoder_sent_length,axis=1)  # it is a list,length is decoder_sent_length, each element is [batch_size,1,embed_size]
        #embedded_query_squeezed = [tf.squeeze(x, axis=1) for x in embedded_query_splitted]   # it is a list,length is decoder_sent_length, each element is [batch_size,embed_size]

        decoder_input_embedded=tf.nn.embedding_lookup(self.Embedding_label,self.decoder_input) #[batch_size,self.decoder_sent_length,embed_size]
        decoder_input_splitted = tf.split(decoder_input_embedded, self.decoder_sent_length,axis=1)  # it is a list,length is decoder_sent_length, each element is [batch_size,1,embed_size]
        decoder_input_squeezed = [tf.squeeze(x, axis=1) for x in decoder_input_splitted]  # it is a list,length is decoder_sent_length, each element is [batch_size,embed_size]

        decoder_inputs = [tf.concat([x1, x2], axis=1) for x1, x2 in zip(decoder_input_squeezed, embedded_query_squeezed)] ## it is a list,length is decoder_sent_length, each element is [batch_size,embed_size*2]

        initial_state=tf.nn.tanh(tf.matmul(decoder_inputs[0],self.W_initial_state)+self.b_initial_state) #hidden_state_backward_list[0].initial_state:[batch_size,hidden_size*2]. TODO this is follow paper's style.
        cell=self.gru_cell_decoder #this is a special cell. because it beside previous hidden state, current input, it also has a context vecotor, which represent attention result.

        decoder_output,_=self.rnn_decoder_with_attention_ptr_network(decoder_inputs, initial_state, cell, self.is_training, attention_states) # A list.length:self.sequence_length.each element is:[batch_size x output_size]
        decoder_output=tf.stack(decoder_output,axis=1) #decoder_output:[batch_size,decode_sequence_length,input_sequence_length]
        #decoder_output=tf.reshape(decoder_output,shape=(-1,self.sequence_length)) #decoder_output:[batch_size*decoder_sent_length,sequence_length]

        #with tf.name_scope("dropout"):
        #    decoder_output = tf.nn.dropout(decoder_output,keep_prob=self.dropout_keep_prob)  # shape:[batch_size*decoder_sent_length,sequence_length]
        # 4. get logits
        #with tf.name_scope("output"):
        #    logits = tf.matmul(decoder_output, self.W_projection) + self.b_projection  # logits shape:[batch_size*decoder_sent_length,self.sequence_length]==tf.matmul([batch_size*decoder_sent_length,sequence_length],[sequence_length,self.sequence_length])
        #    logits=tf.reshape(logits,shape=(self.batch_size,self.decoder_sent_length,self.sequence_length)) #logits shape:[batch_size,decoder_sent_length,self.sequence_length]
        logits=decoder_output
        return logits

    def rnn_decoder_with_attention_ptr_network(self,decoder_inputs, initial_state, cell, is_training, attention_states,scope='ptr_decoder'):  # 3D Tensor [batch_size x attn_length x attn_size]
        """RNN decoder for the sequence-to-sequence model.
        Args:
            decoder_inputs: A list of 2D Tensors [batch_size x input_size].it is decoder input.
            initial_state: 2D Tensor with shape [batch_size x cell.state_size].it is the encoded vector of input sentences, which represent 'thought vector'
            cell: core_rnn_cell.RNNCell defining the cell function and size.
            is_training:  If it is not training, decoder_input will be ignored, and will use generated token.
            (loop_function,removed): If not None, this function will be applied to the i-th output
                in order to generate the i+1-st input, and decoder_inputs will be ignored,
                except for the first element ("GO" symbol). This can be used for decoding,
                but also for training to emulate http://arxiv.org/abs/1506.03099.
                Signature -- loop_function(prev, i) = next
                    * prev is a 2D Tensor of shape [batch_size x output_size],
                    * i is an integer, the step number (when advanced control is needed),
                    * next is a 2D Tensor of shape [batch_size x input_size].
            attention_states: 3D Tensor [batch_size x attn_length x attn_size].it is represent input X.
            scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
        Returns:
            A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x output_size] containing generated outputs.
            state: The state of each cell at the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
                (Note that in some cases, like basic RNN cell or GRU cell, outputs and
                states can be the same. They are different for LSTM cells though.)
        """
        with tf.variable_scope("rnn_decoder"):#scope or "rnn_decoder"
            print("rnn_decoder_with_attention started...")
            state = initial_state  # [batch_size x cell.state_size].
            _, hidden_size = state.get_shape().as_list()  # 200
            attention_states_original = attention_states  # it is represent input X.
            batch_size, sequence_length, _ = attention_states.get_shape().as_list()
            outputs = []
            prev = None
            for i, inp in enumerate(decoder_inputs):  #sentence_length个[batch_size x input_size]
                if  not is_training and prev is not None:
                    with tf.variable_scope("loop_function", reuse=True):
                        # inp = loop_function(prev, i)
                        prev_symbol = tf.argmax(prev, 1)  # [batch_size]
                        prev_symbol = prev_symbol[0]  # only care about one when predict, since batch_size will be 1.
                        inp = attention_states[:, prev_symbol, :]
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                # 1. ATTENTION get logits of attention for each encoder input. attention_states:[batch_size x attn_length x attn_size]; query=state:[batch_size x cell.state_size]
                query = state
                W_a = tf.get_variable("W_a", shape=[hidden_size, hidden_size], initializer=tf.random_normal_initializer(stddev=0.1))
                query = tf.matmul(query, W_a)  # [batch_size,hidden_size]
                query = tf.expand_dims(query, axis=1)  # [batch_size, 1, hidden_size]
                U_a = tf.get_variable("U_a", shape=[hidden_size, hidden_size],initializer=tf.random_normal_initializer(stddev=0.1))
                U_aa = tf.get_variable("U_aa", shape=[hidden_size])
                attention_states = tf.reshape(attention_states,shape=(-1, hidden_size))  # [batch_size*sentence_length,hidden_size]
                attention_states = tf.matmul(attention_states, U_a)  # [batch_size*sentence_length,hidden_size]
                attention_states = tf.reshape(attention_states, shape=(-1, sequence_length, hidden_size))  # TODO [batch_size,sentence_length,hidden_size]
                # query_expanded:            [batch_size,1,             hidden_size]
                # attention_states_reshaped: [batch_size,sentence_length,hidden_size]
                # query:last state of x
                # attention_states: represent x in 3D
                attention_logits = tf.nn.tanh(query + attention_states + U_aa)  # [batch_size,sentence_length,hidden_size]. additive style

                # 2.get possibility of attention
                attention_logits = tf.reshape(attention_logits, shape=(-1, hidden_size))  # batch_size*sequence_length [batch_size*sentence_length,hidden_size]
                V_a = tf.get_variable("V_a", shape=[hidden_size, 1],initializer=tf.random_normal_initializer(stddev=0.1))  # [hidden_size,1]
                attention_logits = tf.matmul(attention_logits,V_a)  # 最终需要的是[batch_size*sentence_length,1]<-----[batch_size*sentence_length,hidden_size],[hidden_size,1]
                attention_logits = tf.reshape(attention_logits, shape=(-1, sequence_length))  # attention_logits:[batch_size,sequence_length]
                attention_logits_max = tf.reduce_max(attention_logits, axis=1, keep_dims=True)  # [batch_size x 1]
                # possibility distribution for each encoder input.it means how much attention or focus for each encoder input
                p_attention = tf.nn.softmax(attention_logits - attention_logits_max)  # [batch_size x sequence_length] #=[batch_size,sequence_length of input]
                output, state = cell(inp, state)  # cell(inp, state,context_vector)
                outputs.append(p_attention)
                if not is_training:
                    prev = p_attention  # [batch_size x sequence_length] #=[batch_size,sequence_length of input]
        print("rnn_decoder_with_attention ended...outputs:")
        return outputs, state


    def loss_seq2seq(self):
        with tf.name_scope("loss"):
            #input: `logits` and `labels` must have the same shape `[batch_size, sequence_length]`
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y_label, logits=self.logits);#losses:[batch_size,self.decoder_sent_length]
            loss_batch=tf.reduce_sum(losses,axis=1)/self.decoder_sent_length #loss_batch:[batch_size]
            loss=tf.reduce_mean(loss_batch)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.l2_lambda
            loss = loss + l2_losses
            return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        self.learning_rate_=learning_rate
        #noise_std_dev = tf.constant(0.3) / (tf.sqrt(tf.cast(tf.constant(1) + self.global_step, tf.float32))) #gradient_noise_scale=noise_std_dev
        train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

    def gru_cell(self, Xt, h_t_minus_1,scope='gru_cell'):
        """
        single step of gru for word level
        :param Xt: Xt:[batch_size,embed_size]
        :param h_t_minus_1:[batch_size,embed_size]
        :return:
        """
        with tf.variable_scope(scope):
            # 1.update gate: decides how much past information is kept and how much new information is added.
            z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z) + tf.matmul(h_t_minus_1,self.U_z) + self.b_z)  # z_t:[batch_size,self.hidden_size]
            # 2.reset gate: controls how much the past state contributes to the candidate state.
            r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r) + tf.matmul(h_t_minus_1,self.U_r) + self.b_r)  # r_t:[batch_size,self.hidden_size]
            # candiate state h_t~
            h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h) +r_t * (tf.matmul(h_t_minus_1, self.U_h)) + self.b_h)  # h_t_candiate:[batch_size,self.hidden_size]
            # new state: a linear combine of pervious hidden state and the current new state h_t~
            h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate  # h_t:[batch_size*num_sentences,hidden_size]
        return h_t

    def gru_cell_decoder(self, Xt, h_t_minus_1):
        """
        single step of gru for word level
        :param Xt: Xt:[batch_size,embed_size]
        :param h_t_minus_1:[batch_size,embed_size]
        :param context_vector. [batch_size,embed_size].this represent the result from attention( weighted sum of input during current decoding step)
        :return:
        """
        # 1.update gate: decides how much past information is kept and how much new information is added.
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z_decoder) + tf.matmul(h_t_minus_1, self.U_z_decoder)  + self.b_z_decoder)  # z_t:[batch_size,self.hidden_size]
        # 2.reset gate: controls how much the past state contributes to the candidate state.
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r_decoder) + tf.matmul(h_t_minus_1, self.U_r_decoder)  + self.b_r_decoder)  # r_t:[batch_size,self.hidden_size]
        # candiate state h_t~
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h_decoder) + r_t * (tf.matmul(h_t_minus_1, self.U_h_decoder))  + self.b_h_decoder)  # h_t_candiate:[batch_size,self.hidden_size]
        # new state: a linear combine of pervious hidden state and the current new state h_t~
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate  # h_t:[batch_size*num_sentences,hidden_size]
        return h_t, h_t

    # forward gru for first level: word levels
    def gru_forward(self, embedded_words,gru_cell,scope='gru_forward', reverse=False,):
        """
        :param embedded_words:[None,sequence_length, self.embed_size]
        :return:forward hidden state: a list.length is sentence_length, each element is [batch_size,hidden_size]
        """
        with tf.variable_scope(scope):
            # split embedded_words
            sequence_length=embedded_words.get_shape().as_list()[1]
            embedded_words_splitted = tf.split(embedded_words, sequence_length,axis=1)  # it is a list,length is sentence_length, each element is [batch_size,1,embed_size]
            embedded_words_squeeze = [tf.squeeze(x, axis=1) for x in embedded_words_splitted]  # it is a list,length is sentence_length, each element is [batch_size,embed_size]
            h_t = tf.ones((self.batch_size,self.hidden_size))
            h_t_list = []
            if reverse:
                embedded_words_squeeze.reverse()
            for time_step, Xt in enumerate(embedded_words_squeeze):  # Xt: [batch_size,embed_size]
                h_t = gru_cell(Xt,h_t,scope=scope) #h_t:[batch_size,embed_size]<------Xt:[batch_size,embed_size];h_t:[batch_size,embed_size]
                h_t_list.append(h_t)
            if reverse:
                h_t_list.reverse()
        return h_t_list  # a list,length is sentence_length, each element is [batch_size,hidden_size]

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("decoder_init_state"):
            self.W_initial_state = tf.get_variable("W_initial_state", shape=[self.hidden_size*3, self.hidden_size*2], initializer=self.initializer)
            self.b_initial_state = tf.get_variable("b_initial_state", shape=[self.hidden_size*2])
        with tf.name_scope("embedding_projection"):  # embedding matrix
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],initializer=self.initializer)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.Embedding_label = tf.get_variable("Embedding_label", shape=[self.vocab_size, self.embed_size*2],dtype=tf.float32) #,self.num_classes
            self.W_projection = tf.get_variable("W_projection", shape=[self.sequence_length, self.sequence_length],#num_classes
                                                initializer=self.initializer)  # [embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection", shape=[self.sequence_length]) #num_classes

        # GRU parameters:update gate related
        with tf.name_scope("gru_weights_encoder"):
            self.W_z = tf.get_variable("W_z", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_z = tf.get_variable("U_z", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_z = tf.get_variable("b_z", shape=[self.hidden_size])
            # GRU parameters:reset gate related
            self.W_r = tf.get_variable("W_r", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_r = tf.get_variable("U_r", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_r = tf.get_variable("b_r", shape=[self.hidden_size])

            self.W_h = tf.get_variable("W_h", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_h = tf.get_variable("U_h", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_h = tf.get_variable("b_h", shape=[self.hidden_size])

        with tf.name_scope("gru_weights_decoder"):
            self.W_z_decoder = tf.get_variable("W_z_decoder", shape=[self.embed_size*3, self.hidden_size*2], initializer=self.initializer)
            self.U_z_decoder = tf.get_variable("U_z_decoder", shape=[self.embed_size*2, self.hidden_size*2], initializer=self.initializer)
            self.C_z_decoder = tf.get_variable("C_z_decoder", shape=[self.embed_size * 2, self.hidden_size * 2],initializer=self.initializer) #TODO
            self.b_z_decoder = tf.get_variable("b_z_decoder", shape=[self.hidden_size*2])
            # GRU parameters:reset gate related
            self.W_r_decoder = tf.get_variable("W_r_decoder", shape=[self.embed_size*3, self.hidden_size*2], initializer=self.initializer)
            self.U_r_decoder = tf.get_variable("U_r_decoder", shape=[self.embed_size*2, self.hidden_size*2], initializer=self.initializer)
            self.C_r_decoder = tf.get_variable("C_r_decoder", shape=[self.embed_size * 2, self.hidden_size * 2],initializer=self.initializer) #TODO
            self.b_r_decoder = tf.get_variable("b_r_decoder", shape=[self.hidden_size*2])

            self.W_h_decoder = tf.get_variable("W_h_decoder", shape=[self.embed_size*3, self.hidden_size*2], initializer=self.initializer)
            self.U_h_decoder = tf.get_variable("U_h_decoder", shape=[self.embed_size*2, self.hidden_size*2], initializer=self.initializer)   #TODO
            self.C_h_decoder = tf.get_variable("C_h_decoder", shape=[self.embed_size * 2, self.hidden_size * 2],initializer=self.initializer)
            self.b_h_decoder = tf.get_variable("b_h_decoder", shape=[self.hidden_size*2])

        with tf.name_scope("transform"):
            self.H= tf.get_variable("H", shape=[self.hidden_size*2, self.hidden_size*2], initializer=self.initializer)

        with tf.name_scope("full_connected"):
            self.W_fc=tf.get_variable("W_fc",shape=[self.hidden_size*2,self.hidden_size])
            self.a_fc=tf.get_variable("a_fc",shape=[self.hidden_size])

    def prelu(self,features, scope=None):  # scope=None
        with tf.variable_scope(scope, 'PReLU', initializer=self.initializer):
            alpha = tf.get_variable('alpha', features.get_shape().as_list()[1:])
            pos = tf.nn.relu(features)
            neg = alpha * (features - tf.abs(features)) * 0.5
            return pos + neg

# test started: learn to sort nature number. for example, give a list[3,5,2,5,6,1], it will output:[5,2,0,3,1,4],which represent the index of the number in the list,sort by ascending order

#train()-->predict()

start_token_index = 99
end_token_index=7

sequence_length=20

def train_batch():
    # below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
    #num_classes = 9+2 #additional two classes:one is for _GO, another is for _END
    learning_rate = 0.001
    batch_size = 64
    decay_steps = 1000
    decay_rate = 0.98
    vocab_size = 300
    embed_size = 100 #100
    hidden_size = 100
    is_training = True
    dropout_keep_prob =0.0 # 0.5  # 0.5 #num_sentences
    decoder_sent_length=sequence_length+1 #6
    l2_lambda=0.0001
    model = dynamic_pointer_net(learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                                    vocab_size, embed_size,hidden_size, is_training,decoder_sent_length=decoder_sent_length,l2_lambda=l2_lambda)
    ckpt_dir = 'checkpoint_dynamic_pointer_net/dummy_test/'
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if os.path.exists(ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())

        for i in range(1500):
            input_x, decoder_input, input_y_label,decoder_input_reverse,input_y_label_reverse,decoder_input_max,input_y_label_max=get_unique_labels_batch(batch_size,sequence_length=sequence_length) #o.k. a 2-D list
            if i%3==0:
                sorting=[[101]*decoder_sent_length]*batch_size #sorting as ascending order(升序)
            elif i%3==1:
                sorting =[[102]*decoder_sent_length]*batch_size  # sorting as ascending order（降序）
                decoder_input=decoder_input_reverse
                input_y_label=input_y_label_reverse
            elif i%3==2:
                sorting = [[103] * decoder_sent_length] * batch_size  # get max value
                decoder_input=decoder_input_max
                input_y_label=input_y_label_max

            query=np.array(sorting,dtype=np.int32) # query
            loss, acc, predict, W_projection_value, _ = sess.run([model.loss_val, model.accuracy, model.predictions, model.W_projection, model.train_op],
                                                     feed_dict={model.input_x:input_x,model.query:query,model.decoder_input:decoder_input, model.input_y_label: input_y_label,
                                                                model.dropout_keep_prob: dropout_keep_prob})
            print(i,"loss:", loss, "acc:", acc);#print( "label_list_original as input x:",label_list_original,";input_y_label:", input_y_label, "prediction:", predict)

            if i % 500 == 0:
                save_path = ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=i * 500)

def predict():
    print("predict started.")
    learning_rate = 0.0001
    batch_size = 1
    decay_steps = 1000
    decay_rate = 0.9
    vocab_size = 300
    embed_size = 100 #100
    hidden_size = 100
    is_training = True
    dropout_keep_prob = 0.5  # 0.5 #num_sentences
    decoder_sent_length=sequence_length+1
    l2_lambda=0.0001
    #start_token_index = 100
    model = dynamic_pointer_net(learning_rate, batch_size, decay_steps, decay_rate, sequence_length, vocab_size,
                                    embed_size, hidden_size, is_training, decoder_sent_length=decoder_sent_length,l2_lambda=l2_lambda)
    ckpt_dir = 'checkpoint_dynamic_pointer_net/dummy_test/'
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("going to restore checkpoint.")
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        for i in range(100):
            label_list=[7, 14, 11, 19, 18, 8, 12, 9, 13, 3, 5, 1, 6, 17, 16, 15, 20, 4, 10, 2] #get_unique_labels(sequence_length=sequence_length) # length is 5.
            input_x = np.array([label_list],dtype=np.int32) #[2,3,4,5,6]
            if i%3==0:
                sorting=[101]*decoder_sent_length   #sorting as ascending order(升序)
            elif i%3==1:
                sorting =[102]*decoder_sent_length  # sorting as ascending order（降序）
            elif i % 3 == 2:
                sorting = [103] * decoder_sent_length

            query=np.array([sorting],dtype=np.int32) # query
            label_list_original=copy.deepcopy(label_list)
            decoder_input=np.array([[start_token_index]*decoder_sent_length],dtype=np.int32) #[[0,2,3,4,5,6]]
            predict, = sess.run(model.predictions,feed_dict={model.input_x: input_x, model.query:query,model.decoder_input: decoder_input,
                                                             model.dropout_keep_prob: dropout_keep_prob})

            label_target = np.argsort(label_list)
            label_target = list(label_target)
            if i % 3 ==1:
                label_target.reverse()
            elif i%3==2:
                #label_target = [1] * len(label_list)
                #max_value_index = np.argmax(label_list)
                #label_target[max_value_index] = 0
                label_target = [i for i in range(len(label_list))]  # TODO #[1]*len(labels)
                max_value_index = np.argmax(label_list)
                label_target[max_value_index] = 0
            input_y_label = np.array([label_target ], dtype=np.int32)  # [[2,3,4,5,6,1]]
            print(i,  "label_list_original as input x:", label_list_original,";input_y_label:", input_y_label, "prediction:", predict) #



def get_unique_labels_batch(batch_size,sequence_length=8):
    #print("get_unique_labels_batch.sequence_length:",sequence_length)
    x=[]
    decoder_input=[]
    input_y_label=[]
    decoder_input_reverse=[]
    input_y_label_reverse=[]
    decoder_input_max=[]
    input_y_label_max=[]
    for i in range(batch_size):
        labels=get_unique_labels(sequence_length=sequence_length)
        #print("labels:",labels)
        x.append(labels)

        sequence_target_index=list(np.argsort(labels))
        decoder_input.append([start_token_index]+sequence_target_index)
        input_y_label.append(sequence_target_index+[end_token_index])

        sequence_target_index.reverse()
        decoder_input_reverse.append([start_token_index]+sequence_target_index)
        input_y_label_reverse.append(sequence_target_index + [end_token_index])

        max_value_list=[i for i in range(len(labels))] #TODO #[1]*len(labels)
        max_value_index=np.argmax(labels)
        max_value_list[max_value_index]=0
        #print("max.labels:",labels,";max_value_list:",max_value_list)
        decoder_input_max.append([start_token_index]+max_value_list)
        input_y_label_max.append(max_value_list+[end_token_index])
        #print([start_token_index]+max_value_list,"--->",max_value_list+[end_token_index])
    return np.array(x),np.array(decoder_input),np.array(input_y_label),np.array(decoder_input_reverse),np.array(input_y_label_reverse),np.array(decoder_input_max),np.array(input_y_label_max)

def get_unique_labels(sequence_length=8):
    #print("get_unique_labels.sequence_length:",sequence_length)
    x=[x+1 for x in range(sequence_length)]
    #x = [2, 3, 4, 5, 6,7,8,9] #x = [2, 3, 4, 5, 6]
    random.shuffle(x)
    return x

#1.train the model
#train_batch()
#2.make a prediction based on the learned model.
predict()