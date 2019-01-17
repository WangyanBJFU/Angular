import tensorflow as tf
import tflearn

def L2_Sofrmax_Loss(embeddings, labels, num_classes, margin=1):
    with tf.variable_scope("softmax"):
        weights = tf.get_variable(name='embedding_weights',
                                  shape=[embeddings.get_shape().as_list()[-1], num_classes],
                                  initializer=tf.contrib.layers.xavier_initializer())
        # 初始化weights，shape=[embeddings的列，num_classes]
        # 时候使每一层梯度大小都差不多相同
        weights = tf.nn.l2_normalize(weights, dim=1)
        # 按列进行L2范化 范化=原值/norm(所在列)=原值/根号(列上所有值的平方和)

        # caculating the cos value of angles between embeddings and weights
        original_logits = tf.matmul(embeddings, weights) #[行,列]×[列,num_classes] = [行,num_classes]

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=original_logits))
        pred_prob = tf.nn.softmax(logits=original_logits)

        return pred_prob, loss

def add_fc_layer(inputs, in_size, out_size, name, activation_function=None):
    with tf.variable_scope(name) as scope:
        w = tf.Variable(tf.random_normal([in_size, out_size]))
        b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.nn.xw_plus_b(inputs, w, b, name=scope.name)
        if activation_function is None:
            output = Wx_plus_b
        else:
            output = activation_function(Wx_plus_b)
        return output

def add_conv_layer(x, kernel_height, kernel_width, strideX, strideY, feature_num, name, padding = 'SAME'):

    channel = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('w', shape=[kernel_height, kernel_width, channel, feature_num])
        b = tf.get_variable('b', shape=[feature_num])
        feature_map = tf.nn.conv2d(x, w, strides=[1, strideX, strideY, 1], padding=padding)
        output = tf.nn.bias_add(feature_map, b, name=scope)
        # return tf.nn.relu(output, feature_map.get_shape().as_list(), name=scope.name)
        return tf.nn.relu(output)
    
def add_maxpool_layer(x, kernel_height, kernel_width, strideX, strideY, name, padding = 'SAME'):
    with tf.variable_scope(name) as scope:
        return tf.nn.max_pool(x, ksize=[1, kernel_height, kernel_width, 1],
                              strides=[1, strideX, strideY, 1],
                              padding=padding, name=scope.name)

def add_dropout_layer(x, keep_prob, name=None):
    return tf.nn.dropout(x, keep_prob, name)

def buildCNN(data_input):
    x = tflearn.conv_2d(data_input, 32, 3, strides = 1, activation='prelu', weights_init = 'xavier')
    x = tflearn.conv_2d(x, 32, 3, strides = 2, activation='prelu', weights_init = 'xavier')
    x = tflearn.conv_2d(x, 64, 3, strides = 1, activation='prelu', weights_init = 'xavier')
    x = tflearn.conv_2d(x, 64, 3, strides = 2, activation='prelu', weights_init = 'xavier')
    x = tflearn.conv_2d(x, 128, 3, strides = 1, activation='prelu', weights_init = 'xavier')
    x = tflearn.conv_2d(x, 128, 3, strides = 2, activation='prelu', weights_init = 'xavier')
    x = tflearn.flatten(x)
    embeddings = tflearn.fully_connected(x, 2, weights_init = 'xavier')
    y = add_fc_layer(x, 128, 10, "y", "relu")
    return embeddings
