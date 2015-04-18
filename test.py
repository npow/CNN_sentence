import cPickle
import lasagne
import numpy as np
import theano
import theano.tensor as T
import sys

def train_conv_net(datasets,
                   U,
                   img_w=300, 
                   filter_hs=[3,4,5],
                   hidden_units=[100,2], 
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=25, 
                   batch_size=50, 
                   lr_decay = 0.95,
                   conv_non_linear=lasagne.nonlinearities.rectify,
                   sqr_norm_lim=9,
                   non_static=True):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """    
    img_h = len(datasets[0][0])-1  
    print "img_h: ", img_h
    conv_layers = []
    input_layers = []
    for i in xrange(len(filter_hs)):
        l_in = lasagne.layers.InputLayer(shape=(batch_size, 1, img_h, img_w))
        conv_layer = lasagne.layers.Conv2DLayer(
                l_in,
                num_filters=hidden_units[0],
                filter_size=(filter_hs[i],img_w),
                strides=(1,1),
                nonlinearity=lasagne.nonlinearities.rectify
                )
        pool_layer = lasagne.layers.MaxPool2DLayer(
                conv_layer,
                ds=(img_h-filter_hs[i]+1, 1)
                )
        conv_layers.append(pool_layer)
        input_layers.append(l_in)

    l_hidden1 = lasagne.layers.ConcatLayer(conv_layers)
    l_hidden1 = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)
    l_out = lasagne.layers.DenseLayer(l_hidden1, num_units=hidden_units[1], nonlinearity=lasagne.nonlinearities.softmax)

    index = T.lscalar()
    x = T.imatrix('x')   
    y = T.ivector('y')
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w, dtype=theano.config.floatX)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))])
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))                                  
    
    objective = lasagne.objectives.Objective(l_out, loss_function=lasagne.objectives.categorical_crossentropy)
    cost = objective.get_loss(layer0_input, target=y)
    params = lasagne.layers.get_all_params(l_out)
    updates = lasagne.updates.adadelta(cost, params, learning_rate=1.0, rho=lr_decay)
    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random)
    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])   
        extra_data = train_set[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)
    else:
        new_data = datasets[0]
    new_data = np.random.permutation(new_data)
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
    #divide train set into train/val sets 
    test_set_x = datasets[1][:,:img_h] 
    test_set_y = np.asarray(datasets[1][:,-1],"int32")
    train_set = new_data[:n_train_batches*batch_size,:]
    val_set = new_data[n_train_batches*batch_size:,:]     
    train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))
    val_set_x, val_set_y = shared_dataset((val_set[:,:img_h],val_set[:,-1]))
    n_val_batches = n_batches - n_train_batches

    pred = T.argmax(l_out.get_output((layer0_input), deterministic=True), axis=1)    
    errors = T.mean(T.neq(pred, y), dtype=theano.config.floatX)
    val_model = theano.function([index], errors,
         givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
            y: val_set_y[index * batch_size: (index + 1) * batch_size]})
            
    #compile theano functions to get train/val/test errors
    test_model = theano.function([index], errors,
             givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]})               
    train_model = theano.function([index], cost, updates=updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]})     

    test_error = T.mean(T.neq(pred, y))
    test_model_all = theano.function([x,y], test_error)   
    
    #start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0    
    while (epoch < n_epochs):        
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)  
                set_zero(zero_vec)
        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)                        
        print('epoch %i, train perf %f %%, val perf %f' % (epoch, train_perf * 100., val_perf*100.))
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            print test_set_x.shape, test_set_y.shape
            print test_set_x[0].shape, test_set_x[1].shape
            test_loss = test_model_all(test_set_x,test_set_y)        
            test_perf = 1- test_loss         
    return test_perf

def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return T.cast(shared_x, 'int32'), T.cast(shared_y, 'int32')

def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)   
        sent.append(rev["y"])
        if rev["split"]==cv:            
            test.append(sent)        
        else:  
            train.append(sent)   
    train = np.array(train,dtype='int32')
    test = np.array(test,dtype='int32')
    return [train, test]     
  
   
if __name__=="__main__":
    print "loading data...",
    x = cPickle.load(open("mr.p","rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1].astype(theano.config.floatX), x[2].astype(theano.config.floatX), x[3], x[4]
    print "data loaded!"
    mode= "-static"#sys.argv[1]
    word_vectors = "-word2vec"#sys.argv[2]
    if mode=="-nonstatic":
        print "model architecture: CNN-non-static"
        non_static=True
    elif mode=="-static":
        print "model architecture: CNN-static"
        non_static=False
    execfile("conv_net_classes.py")    
    if word_vectors=="-rand":
        print "using: random vectors"
        U = W2
    elif word_vectors=="-word2vec":
        print "using: word2vec vectors"
        U = W
    results = []
    r = range(0,10)    
    for i in r:
        datasets = make_idx_data_cv(revs, word_idx_map, i, max_l=56,k=300, filter_h=5)
        print type(datasets[1]), datasets[1].dtype, datasets[1].shape
        perf = train_conv_net(datasets,
                              U,
                              lr_decay=0.95,
                              filter_hs=[3,4,5],
                              hidden_units=[100,2], 
                              shuffle_batch=True, 
                              n_epochs=25, 
                              sqr_norm_lim=9,
                              non_static=non_static,
                              batch_size=50,
                              dropout_rate=[0.5])
        print "cv: " + str(i) + ", perf: " + str(perf)
        results.append(perf)  
    print str(np.mean(results))
