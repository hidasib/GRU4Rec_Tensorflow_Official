import numpy as np
import pandas as pd
import time
import tensorflow as tf
import datetime as dt
import joblib

class SampleCache:
    def __init__(self, n_sample, sample_cache_max_size, distr):
        self.n_sample = n_sample
        self.generate_length = sample_cache_max_size // n_sample if n_sample > 0 else 0
        self.distr = distr
        self._refresh()
        print('Created sample store with {} batches of samples (type=GPU)'.format(self.generate_length))
    def _refresh(self):
        if self.n_sample <= 0: return
        self.neg_samples = tf.searchsorted(self.distr, tf.random.uniform([self.generate_length * self.n_sample]))
        self.neg_samples = tf.reshape(self.neg_samples, (self.generate_length, self.n_sample))
        self.sample_pointer = 0
    def get_sample(self):
        if self.sample_pointer >= self.generate_length:
            self._refresh()
        sample = self.neg_samples[self.sample_pointer]
        self.sample_pointer += 1
        return sample

class SessionDataIterator:
    def __init__(self, data, batch_size, n_sample=0, sample_alpha=0.75, sample_cache_max_size=10000000, item_key='ItemId', session_key='SessionId', time_key='Time', session_order='time', itemidmap=None):
        self.batch_size = batch_size
        if itemidmap is None:
            itemids = data[item_key].unique()
            self.n_items = len(itemids)
            self.itemidmap = pd.Series(data=np.arange(self.n_items, dtype='int32'), index=itemids, name='ItemIdx')
        else:
            print('Using existing item ID map')
            self.itemidmap = itemidmap
            self.n_items = len(itemidmap)
            in_mask = data[item_key].isin(itemidmap.index.values)
            n_not_in = (~in_mask).sum()
            if n_not_in > 0:
                #print('{} rows of the data contain unknown items and will be filtered'.format(n_not_in))
                data = data.drop(data.index[~in_mask])
        self.sort_if_needed(data, [session_key, time_key])
        self.offset_sessions = self.compute_offset(data, session_key)
        if session_order == 'time':
            self.session_idx_arr = np.argsort(data.groupby(session_key)[time_key].min().values)
        else:
            self.session_idx_arr = np.arange(len(self.offset_sessions) - 1)
        self.data_items = self.itemidmap[data[item_key].values].values
        if n_sample > 0:
            pop = data.groupby(item_key).size()
            pop = pop[self.itemidmap.index.values].values**sample_alpha
            pop = pop.cumsum() / pop.sum()
            pop[-1] = 1
            distr = tf.constant(pop.astype('float32'))
            self.sample_cache = SampleCache(n_sample, sample_cache_max_size, distr)
    
    def sort_if_needed(self, data, columns, any_order_first_dim=False):
        is_sorted = True
        neq_masks = []
        for i, col in enumerate(columns):
            dcol = data[col]
            neq_masks.append(dcol.values[1:]!=dcol.values[:-1])
            if i == 0:
                if any_order_first_dim:
                    is_sorted = is_sorted and (dcol.nunique() == neq_masks[0].sum() + 1)
                else:
                    is_sorted = is_sorted and np.all(dcol.values[1:] >= dcol.values[:-1])
            else:
                is_sorted = is_sorted and np.all(neq_masks[i - 1] | (dcol.values[1:] >= dcol.values[:-1]))
            if not is_sorted:
                break
        if is_sorted:
            print('The dataframe is already sorted by {}'.format(', '.join(columns)))
        else:
            print('The dataframe is not sorted by {}, sorting now'.format(col))
            t0 = time.time()
            data.sort_values(columns, inplace=True)
            t1 = time.time()
            print('Data is sorted in {:.2f}'.format(t1 - t0))

    def compute_offset(self, data, column):
        offset = np.zeros(data[column].nunique() + 1, dtype=np.int32)
        offset[1:] = data.groupby(column).size().cumsum()
        return offset
    
    def __call__(self, enable_neg_samples, reset_hook=None):
        batch_size = self.batch_size
        iters = np.arange(batch_size)
        maxiter = iters.max()
        start = self.offset_sessions[self.session_idx_arr[iters]]
        end = self.offset_sessions[self.session_idx_arr[iters]+1]
        finished = False
        valid_mask = np.ones(batch_size, dtype='bool')
        n_valid = self.batch_size
        while not finished:
            minlen = (end - start).min()
            out_idx = self.data_items[start]
            for i in range(minlen - 1):
                in_idx = out_idx
                out_idx = self.data_items[start + i + 1]
                reset_after = (start + i + 1 == end - 1)
                if enable_neg_samples and hasattr(self, 'sample_cache'):
                    sample = self.sample_cache.get_sample()
                    y = tf.concat([tf.constant(out_idx), sample], axis=0)
                else:
                    y = tf.constant(out_idx)
                yield tf.constant(in_idx), y
            start = start + minlen - 1
            finished_mask = (end - start <= 1)
            n_finished = finished_mask.sum()
            iters[finished_mask] = maxiter + np.arange(1, n_finished + 1)
            maxiter += n_finished
            valid_mask = (iters < len(self.offset_sessions) - 1)
            n_valid = valid_mask.sum()
            if n_valid == 0:
                finished = True
                break
            mask = finished_mask & valid_mask
            sessions = self.session_idx_arr[iters[mask]]
            start[mask] = self.offset_sessions[sessions]
            end[mask] = self.offset_sessions[sessions + 1]
            iters = iters[valid_mask]
            start = start[valid_mask]
            end = end[valid_mask]
            if reset_hook is not None:
                finished = reset_hook(n_valid, finished_mask, valid_mask)

class AdagradM(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.05, momentum=0.0, epsilon=1e-6, weight_decay=None, clipnorm=None, clipvalue=None, global_clipnorm=None, 
                 use_ema=False, ema_momentum=0.99, ema_overwrite_frequency=None, jit_compile=False, name="AdagradM", **kwargs):
        super().__init__(weight_decay=weight_decay, clipnorm=clipnorm, clipvalue=clipvalue, global_clipnorm=global_clipnorm, use_ema=use_ema, ema_momentum=ema_momentum, ema_overwrite_frequency=ema_overwrite_frequency, jit_compile=jit_compile, name=name, **kwargs)
        assert momentum >= 0
        assert epsilon > 0
        self.epsilon = epsilon
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.momentum = momentum
    
    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._acc = []
        self._mom = []
        for var in var_list:
            self._acc.append(self.add_variable_from_reference(var, "acc"))
            if self.momentum > 0:
                self._mom.append(self.add_variable_from_reference(var, "mom"))

    def update_step(self, grad, variable):
        learning_rate = tf.cast(self._learning_rate, variable.dtype)
        momentum = tf.cast(self.momentum, variable.dtype)

        var_key = self._var_key(variable)
        acc = self._acc[self._index_dict[var_key]]
        if self.momentum:
            mom = self._mom[self._index_dict[var_key]]

        if isinstance(grad, tf.IndexedSlices):
            # Sparse gradients.
            acc.scatter_add(tf.IndexedSlices(grad.values * grad.values, grad.indices))
            scaled_grad = grad.values / tf.sqrt(tf.gather(acc, indices=grad.indices) + self.epsilon)
            if self.momentum:
                #Incorrect (duplicate indices) but faster
                #velocity = momentum * tf.gather(mom, indices=grad.indices) - learning_rate * scaled_grad
                #mom.scatter_update(tf.IndexedSlices(velocity, grad.indices))
                #variable.scatter_add(tf.IndexedSlices(velocity, grad.indices))
                #Correct but slower
                mom.scatter_add(tf.IndexedSlices(-learning_rate * scaled_grad / momentum, grad.indices))
                velocity = momentum * tf.gather(mom, indices=grad.indices)
                mom.scatter_update(tf.IndexedSlices(velocity, grad.indices))
                variable.scatter_add(tf.IndexedSlices(velocity, grad.indices))
            else:
                variable.scatter_add(tf.IndexedSlices(-learning_rate * scaled_grad, grad.indices))
        else:
            # Dense gradients.
            acc.assign_add(grad * grad)
            scaled_grad = grad / tf.sqrt(acc + self.epsilon)
            if self.momentum > 0:
                mom.assign(momentum * mom - learning_rate * scaled_grad)
                variable.assign_add(mom)
            else:
                variable.assign_sub(learning_rate * scaled_grad)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
                "momentum": self.momentum,
                "epsilon": self.epsilon,
            }
        )
        return config
    
    def get_state(self):
        state = dict()
        if hasattr(self, '_acc'):
            state['_acc'] = []
            for a in self._acc:
                state['_acc'].append(a.numpy())
        if hasattr(self, '_mom'):
            state['_mom'] = []
            for m in self._mom:
                state['_mom'].append(m.numpy())
        return state
    
    def set_state(self, state):
        if '_acc' in state:
            self._acc = []
            for a in state['_acc']:
                self._acc.append(tf.Variable(a))
        if '_mom' in state:
            self._mom = []
            for m in state['_mom']:
                self._mom.append(tf.Variable(m))

def init_weights_2D(dim0, dim1, dim0_scale=1, dim1_scale=1):
    sigma = np.sqrt(6.0 / (dim0 / dim0_scale + dim1 / dim1_scale))
    return tf.random.uniform((dim0, dim1), minval=-sigma, maxval=sigma)

class Embedding(tf.Module):
    def __init__(self, input_dim, output_size, init_with_zero=False, name=None):
        super().__init__(name=name)
        if init_with_zero:
            self.E = tf.Variable(np.zeros((input_dim, output_size), dtype='float32'))
        else:
            self.E = tf.Variable(init_weights_2D(input_dim, output_size))
        self.EX = None
    def __call__(self, X):
        EX = tf.gather(self.E, X)
        self.EX = tf.IndexedSlices(EX, X)
        return EX
    @property
    def variables(self):
        return (self.EX) if self.EX is not None else ()
    @property
    def trainable_variables(self):
        return (self.EX) if self.EX is not None else ()

class GRUEmbedding(tf.Module):
    def __init__(self, input_dim, output_size, name=None):
        super().__init__(name=name)
        self.Wx0 = Embedding(input_dim, output_size * 3)
        self.Wh0 = tf.Variable(init_weights_2D(output_size, output_size * 1, dim1_scale=1))
        self.Wrz0 = tf.Variable(init_weights_2D(output_size, output_size * 2, dim1_scale=2))
        self.Bh0 = tf.Variable(tf.zeros((1, output_size * 3)))
    def __call__(self, X, H):
        Vx = self.Wx0(X) + self.Bh0
        Vrz = tf.matmul(H, self.Wrz0)
        vx_x, vx_r, vx_z = tf.split(Vx, 3, axis=1)
        vh_r, vh_z = tf.split(Vrz, 2, axis=1)
        r = tf.sigmoid(vx_r + vh_r)
        z = tf.sigmoid(vx_z + vh_z)
        h = tf.tanh(tf.matmul(H * r, self.Wh0) + vx_x)
        h = (1.0 - z) * H + z * h
        return h

class GRULayer(tf.Module):
    def __init__(self, input_dim, output_size, name=None):
        super().__init__(name=name)
        self.Wx0 = tf.Variable(init_weights_2D(input_dim, output_size * 3, dim1_scale=3))
        self.Wh0 = tf.Variable(init_weights_2D(output_size, output_size * 1, dim1_scale=1))
        self.Wrz0 = tf.Variable(init_weights_2D(output_size, output_size * 2, dim1_scale=2))
        self.Bh0 = tf.Variable(tf.zeros((1, output_size * 3)))
    def __call__(self, X, H):
        Vx = tf.matmul(X, self.Wx0) + self.Bh0
        Vrz = tf.matmul(H, self.Wrz0)
        vx_x, vx_r, vx_z = tf.split(Vx, 3, axis=1)
        vh_r, vh_z = tf.split(Vrz, 2, axis=1)
        r = tf.sigmoid(vx_r + vh_r)
        z = tf.sigmoid(vx_z + vh_z)
        h = tf.tanh(tf.matmul(H * r, self.Wh0) + vx_x)
        h = (1.0 - z) * H + z * h
        return h

class GRU4RecModel(tf.Module):
    def __init__(self, n_items, layers=[100], dropout_p_embed=0.0, dropout_p_hidden=0.0, embedding=0, constrained_embedding=True, name=None):
        super().__init__(name=name)
        self.n_items = n_items
        self.layers = layers
        self.dropout_p_embed = dropout_p_embed
        self.dropout_p_hidden = dropout_p_hidden
        self.embedding = embedding
        self.constrained_embedding = constrained_embedding
        self.start = 0
        if constrained_embedding:
            n_input = layers[-1]
        elif embedding:
            self.E = Embedding(n_items, embedding)
            n_input = embedding
        else:
            self.GE = GRUEmbedding(n_items, layers[0])
            n_input = n_items
            self.start = 1
        self.DE = tf.keras.layers.Dropout(dropout_p_embed)
        self.G = []
        self.D = []
        for i in range(self.start, len(layers)):
            self.G.append(GRULayer(layers[i-1] if i > 0 else n_input, layers[i]))
            self.D.append(tf.keras.layers.Dropout(dropout_p_hidden))
        self.Wy = Embedding(n_items, layers[-1])
        self.By = Embedding(n_items, 1, init_with_zero=True)
    def _init_numpy_weights(self, shape):
        sigma = np.sqrt(6.0 / (shape[0] + shape[1]))
        m = np.random.rand(*shape).astype('float32') * 2 * sigma - sigma
        return m
    def _reset_weights_to_compatibility_mode(self):
        np.random.seed(42)
        if self.constrained_embedding:
            n_input = self.layers[-1]
        elif self.embedding:
            n_input = self.embedding
            self.E.E.assign(self._init_numpy_weights((self.n_items, n_input)))
        else:
            n_input = self.n_items
            m = []
            m.append(self._init_numpy_weights((n_input, self.layers[0])))
            m.append(self._init_numpy_weights((n_input, self.layers[0])))
            m.append(self._init_numpy_weights((n_input, self.layers[0])))
            self.GE.Wx0.E.assign(np.hstack(m))
            m2 = []
            m2.append(self._init_numpy_weights((self.layers[0] , self.layers[0])))
            m2.append(self._init_numpy_weights((self.layers[0] , self.layers[0])))
            self.GE.Wrz0.assign(np.hstack(m2))
            self.GE.Wh0.assign(self._init_numpy_weights((self.layers[0] , self.layers[0])))
            self.GE.Bh0.assign(np.zeros((1, self.layers[0]*3), dtype='float32'))
        for i in range(self.start, len(self.layers)):
            m = []
            m.append(self._init_numpy_weights((n_input, self.layers[i])))
            m.append(self._init_numpy_weights((n_input, self.layers[i])))
            m.append(self._init_numpy_weights((n_input, self.layers[i])))
            self.G[i].Wx0.assign(np.hstack(m))
            m2 = []
            m2.append(self._init_numpy_weights((self.layers[i] , self.layers[i])))
            m2.append(self._init_numpy_weights((self.layers[i] , self.layers[i])))
            self.G[i].Wrz0.assign(np.hstack(m2))
            self.G[i].Wh0.assign(self._init_numpy_weights((self.layers[i] , self.layers[i])))
            self.G[i].Bh0.assign(np.zeros((1, self.layers[i]*3), dtype='float32'))
        self.Wy.E.assign(self._init_numpy_weights((self.n_items, self.layers[-1])))
        self.By.E.assign(np.zeros((self.n_items, 1), dtype='float32'))
    def embed_constrained(self, X, Y=None):
        if Y is not None:
            XY = tf.concat([X, Y], axis=0)
            EXY = self.Wy(XY)
            split = X.shape[0]
            E = EXY[:split]
            O = EXY[split:]
            B = self.By(Y)
        else:
            E = self.Wy(X)
            O = self.Wy.E.value()
            B = self.By.E.value()
        return E, O, B
    def embed_separate(self, X, Y=None):
        E = self.E(X)
        if Y is not None:
            O = self.Wy(Y)
            B = self.By(Y)
        else:
            O = self.Wy.E.value()
            B = self.By.E.value()
        return E, O, B
    def embed_gru(self, X, H, Y=None):
        E = self.GE(X, H)
        if Y is not None:
            O = self.Wy(Y)
            B = self.By(Y)
        else:
            O = self.Wy.E.value()
            B = self.By.E.value()
        return E, O, B
    def embed(self, X, H, Y=None, training=False):
        if self.constrained_embedding:
            E, O, B = self.embed_constrained(X, Y)
        elif self.embedding > 0:
            E, O, B = self.embed_separate(X, Y)
        else:
            E, O, B = self.embed_gru(X, H[0], Y)
        E = self.DE(E, training=training)
        return E, O, B
    def hidden_step(self, X, H, training=False):
        H2 = []
        for i in range(self.start, len(self.layers)):
            X = self.G[i](X, H[i])
            X = self.D[i](X, training=training)
            H2.append(X)
        return H2
    def score_items(self, X, O, B):
        O = tf.matmul(X, tf.transpose(O)) + tf.transpose(B)
        return O
    def call(self, X, H, Y, training=False):
        E, O, B = self.embed(X, H, Y, training=training)
        if not (self.constrained_embedding or self.embedding):
            H2 = [E]
        else:
            H2 = []
        Xh = self.hidden_step(E, H, training=training)
        for h2 in Xh:
            H2.append(h2)
        R = self.score_items(H2[-1], O, B)
        return R, H2

class GRU4Rec:
    def __init__(self, layers=[100], loss='cross-entropy', batch_size=64, dropout_p_embed=0.0, dropout_p_hidden=0.0, learning_rate=0.05, momentum=0.0, sample_alpha=0.5, n_sample=2048, embedding=0, constrained_embedding=True, n_epochs=10, bpreg=1.0, elu_param=0.5, logq=0.0):
        self.layers = layers
        self.loss = loss
        self.set_loss_function(loss)
        self.elu_param = elu_param
        self.bpreg = bpreg
        self.logq = logq
        self.batch_size = batch_size
        self.dropout_p_embed = dropout_p_embed
        self.dropout_p_hidden = dropout_p_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.sample_alpha = sample_alpha
        self.n_sample = n_sample
        if embedding == 'layersize':
            self.embedding = self.layers[0]
        else:
            self.embedding = embedding
        self.constrained_embedding = constrained_embedding
        self.n_epochs = n_epochs
    def set_params(self, **kvargs):
        maxk_len = np.max([len(str(x)) for x in kvargs.keys()])
        maxv_len = np.max([len(str(x)) for x in kvargs.values()])
        for k,v in kvargs.items():
            if not hasattr(self, k):
                print('Unkown attribute: {}'.format(k))
                raise NotImplementedError
            else:
                if type(v) == str and type(getattr(self, k)) == list: v = [int(l) for l in v.split('/')]
                if type(v) == str and type(getattr(self, k)) == bool:
                    if v == 'True' or v == '1': v = True
                    elif v == 'False' or v == '0': v = False
                    else:
                        print('Invalid value for boolean parameter: {}'.format(v))
                        raise NotImplementedError
                if k == 'embedding' and v == 'layersize':
                    self.embedding = 'layersize'
                setattr(self, k, type(getattr(self, k))(v))
                if k == 'loss': self.set_loss_function(self.loss)
                print('SET   {}{}TO   {}{}(type: {})'.format(k, ' '*(maxk_len-len(k)+3), getattr(self, k), ' '*(maxv_len-len(str(getattr(self, k)))+3), type(getattr(self, k))))
        if self.embedding == 'layersize':
            self.embedding = self.layers[0]
            print('SET   {}{}TO   {}{}(type: {})'.format('embedding', ' '*(maxk_len-len('embedding')+3), getattr(self, 'embedding'), ' '*(maxv_len-len(str(getattr(self, 'embedding')))+3), type(getattr(self, 'embedding'))))
    def set_loss_function(self, loss):
        if loss == 'cross-entropy': self.loss_function = self.xe_loss_with_softmax
        elif loss == 'bpr-max': self.loss_function = self.bpr_max_loss_with_elu
        else: raise NotImplementedError
    def xe_loss_with_softmax(self, O, Y, M):
        if self.logq > 0:
            Y1, Y2 = tf.split(Y, [M, Y.shape[0] - M], axis=0)
            O = O - self.logq * tf.math.log(tf.concat([tf.gather(self.P0, Y1), tf.gather(self.P0, Y2)**self.sample_alpha], axis=0))
        X = tf.exp(O - tf.reduce_max(O, axis=1, keepdims=True))
        X = X / tf.reduce_sum(X, axis=1, keepdims=True)
        return -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(X) + 1e-24))
    def softmax_neg(self, X):
        hm = 1.0 - tf.eye(*X.shape, dtype=X.dtype)
        X = X * hm
        e_x = tf.exp(X - tf.reduce_max(X, axis=1, keepdims=True)) * hm
        return e_x / tf.reduce_sum(e_x, axis=1, keepdims=True)
    def bpr_max_loss_with_elu(self, O, Y, M):
        if self.elu_param > 0:
            O = tf.keras.activations.elu(O, alpha=self.elu_param)
        softmax_scores = self.softmax_neg(O)
        target_scores = tf.expand_dims(tf.linalg.diag_part(O), 1)
        return tf.reduce_sum(-tf.math.log(tf.reduce_sum(tf.sigmoid(target_scores - O) * softmax_scores, axis=1) + 1e-24) + self.bpreg * tf.reduce_sum((O  ** 2) * softmax_scores, axis=1))
    def fit(self, data, sample_cache_max_size=10000000, compatibility_mode=True, item_key='ItemId', session_key='SessionId', time_key='Time'):
        self.error_during_train = False
        self.data_iterator = SessionDataIterator(data, batch_size=self.batch_size, n_sample=self.n_sample, sample_alpha=self.sample_alpha, sample_cache_max_size=sample_cache_max_size, item_key=item_key, session_key=session_key, time_key=time_key, session_order='time')
        if self.logq and self.loss == 'cross-entropy':
            pop = data.groupby(item_key).size()
            self.P0 = tf.constant(pop[self.data_iterator.itemidmap.index.values].astype('float32'))
        model = GRU4RecModel(self.data_iterator.n_items, self.layers, self.dropout_p_embed, self.dropout_p_hidden, self.embedding, self.constrained_embedding)
        if compatibility_mode:
            model._reset_weights_to_compatibility_mode()
        self.model = model
        self.opt = AdagradM(learning_rate=self.learning_rate, momentum=self.momentum)
        for e in range(self.n_epochs):
            loss, dt, events, batches = self._training_loop()
            print('Epoch{} --> loss: {:.6f} \t({:.2f}s) \t[{:.2f} mb/s | {:.0f} e/s]'.format(e+1, loss, dt, batches/dt, events/dt))
    
    def _training_loop(self):
        loss = []
        n = []
        batches = 0
        H = []
        for i in range(len(self.layers)):
            H.append(tf.zeros((self.batch_size, self.layers[i]), dtype=tf.float32))
        t0 = time.time()
        reset_hook = lambda n_valid, finished_mask, valid_mask: self._adjust_hidden(n_valid, finished_mask, valid_mask, H)
        for in_idx, out_idx in self.data_iterator(enable_neg_samples=(self.n_sample>0), reset_hook=reset_hook):
            L, H2 = self._training_step(in_idx, H, out_idx)
            for i in range(len(H)):
                H[i] = H2[i]
            loss.append(L.numpy())
            batches += 1
            n.append(in_idx.shape[0])
        t1 = time.time()
        loss = np.array(loss)
        n = np.array(n)
        events = n.sum()
        return np.sum(loss * n) / events, t1 - t0, events, batches
    
    @tf.function
    def _training_step(self, X, H, Y):
        with tf.GradientTape() as tape:
            R, H2 = self.model.call(X, H, Y, training=True)
            L = self.loss_function(R, Y, X.shape[0]) / self.batch_size
        grads = tape.gradient(L, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return L, H2
    
    def _adjust_hidden(self, n_valid, finished_mask, valid_mask, H):
        if (self.n_sample == 0) and (n_valid < 2): 
            return True
        for i in range(len(self.layers)):
            H[i] = tf.where(finished_mask.reshape(-1,1), 0, H[i])
        if n_valid < len(valid_mask):
            for i in range(len(H)):
                H[i] = H[i][valid_mask]
        return False
    
    def savemodel(self, path):
        #TF is stupid: the built in save doesn't work well with custom stuff & pickle/dill is unable to serialize some constructs -> falling back to moving weights to numpy and then saving
        #TODO: move model state get/set to model
        self.loss_function = None
        self.model.Wy.E = self.model.Wy.E.numpy()
        self.model.Wy.EX = None
        self.model.By.E = self.model.By.E.numpy()
        self.model.By.EX = None
        for i in range(len(self.model.G)):
            self.model.G[i].Wx0 =  self.model.G[i].Wx0.numpy()
            self.model.G[i].Wrz0 =  self.model.G[i].Wrz0.numpy()
            self.model.G[i].Wh0 =  self.model.G[i].Wh0.numpy()
            self.model.G[i].Bh0 =  self.model.G[i].Bh0.numpy()
        if hasattr(self.model, 'E'):
            self.model.E.E = self.model.E.E.numpy()
            self.model.E.EX = None
        if hasattr(self.model, 'GE'):
            self.model.GE.Wx0.E = self.model.GE.Wx0.E.numpy()
            self.model.GE.Wx0.EX = None
            self.model.GE.Wrz0 = self.model.GE.Wrz0.numpy()
            self.model.GE.Wh0 = self.model.GE.Wh0.numpy()
            self.model.GE.Bh0 = self.model.GE.Bh0.numpy()
        #Saving optimizer state (might not be necessary, no training after loading is supported)
        self.opt_state = self.opt.get_state()
        self.opt_config = self.opt.get_config()
        opt = self.opt
        self.opt = None
        joblib.dump(self, path)
        self.opt = opt
        self.opt_state = None
        self.opt_config = None
        #Putting the weights back to GPU after saving, in case the model is used in the same session
        self.set_loss_function(self.loss)
        self.model.Wy.E = tf.Variable(self.model.Wy.E)
        self.model.By.E = tf.Variable(self.model.By.E)
        for i in range(len(self.model.G)):
            self.model.G[i].Wx0 =  tf.Variable(self.model.G[i].Wx0)
            self.model.G[i].Wrz0 =  tf.Variable(self.model.G[i].Wrz0)
            self.model.G[i].Wh0 =  tf.Variable(self.model.G[i].Wh0)
            self.model.G[i].Bh0 =  tf.Variable(self.model.G[i].Bh0)
        if hasattr(self.model, 'E'):
            self.model.E.E = tf.Variable(self.model.E.E)
        if hasattr(self.model, 'GE'):
            self.model.GE.Wx0.E = tf.Variable(self.model.GE.Wx0.E)
            self.model.GE.Wrz0 = tf.Variable(self.model.GE.Wrz0)
            self.model.GE.Wh0 = tf.Variable(self.model.GE.Wh0)
            self.model.GE.Bh0 = tf.Variable(self.model.GE.Bh0)
    
    @classmethod
    def loadmodel(cls, path):
        gru = joblib.load(path)
        gru.set_loss_function(gru.loss)
        gru.model.Wy.E = tf.Variable(gru.model.Wy.E)
        gru.model.By.E = tf.Variable(gru.model.By.E)
        for i in range(len(gru.model.G)):
            gru.model.G[i].Wx0 =  tf.Variable(gru.model.G[i].Wx0)
            gru.model.G[i].Wrz0 =  tf.Variable(gru.model.G[i].Wrz0)
            gru.model.G[i].Wh0 =  tf.Variable(gru.model.G[i].Wh0)
            gru.model.G[i].Bh0 =  tf.Variable(gru.model.G[i].Bh0)
        if hasattr(gru.model, 'E'):
            gru.model.E.E = tf.Variable(gru.model.E.E)
        if hasattr(gru.model, 'GE'):
            gru.model.GE.Wx0.E = tf.Variable(gru.model.GE.Wx0.E)
            gru.model.GE.Wrz0 = tf.Variable(gru.model.GE.Wrz0)
            gru.model.GE.Wh0 = tf.Variable(gru.model.GE.Wh0)
            gru.model.GE.Bh0 = tf.Variable(gru.model.GE.Bh0)
        if hasattr(gru, 'opt_config'):
            gru.opt = AdagradM.from_config(gru.opt_config)
            if hasattr(gru, 'opt_state'):
                gru.opt.set_state(gru.opt_state)
                gru.opt_state = None
            gru.opt_config = None
        return gru
