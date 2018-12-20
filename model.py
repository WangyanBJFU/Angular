import tensorflow as tf
import tflearn

def L2_Sofrmax_Loss(embeddings, labels, num_classes, margin=1):
    embeddings_norm = tf.norm(embeddings, axis=1) # 将w值进行权值归一化

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

'''
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
'''


def buildVGG19(inputs, embedding_dim = 2):
    """block 1"""
    conv1_1 = add_conv_layer(inputs, 3, 3, 1, 1, 64, "conv1_1" )
    conv1_2 = add_conv_layer(conv1_1, 3, 3, 1, 1, 64, "conv1_2")
    pool1 = add_maxpool_layer(conv1_2, 2, 2, 2, 2, "pool1")

    """block 2"""
    conv2_1 = add_conv_layer(pool1, 3, 3, 1, 1, 128, "conv2_1")
    conv2_2 = add_conv_layer(conv2_1, 3, 3, 1, 1, 128, "conv2_2")
    pool2 = add_maxpool_layer(conv2_2, 2, 2, 2, 2, "pool2")

    """block 3"""
    conv3_1 = add_conv_layer(pool2, 3, 3, 1, 1, 256, "conv3_1")
    conv3_2 = add_conv_layer(conv3_1, 3, 3, 1, 1, 256, "conv3_2")
    conv3_3 = add_conv_layer(conv3_2, 3, 3, 1, 1, 256, "conv3_3")
    conv3_4 = add_conv_layer(conv3_3, 3, 3, 1, 1, 256, "conv3_4")
    pool3 = add_maxpool_layer(conv3_4, 2, 2, 2, 2, "pool3")

    """block 4"""
    conv4_1 = add_conv_layer(pool3, 3, 3, 1, 1, 512, "conv4_1")
    conv4_2 = add_conv_layer(conv4_1, 3, 3, 1, 1, 512, "conv4_2")
    conv4_3 = add_conv_layer(conv4_2, 3, 3, 1, 1, 512, "conv4_3")
    conv4_4 = add_conv_layer(conv4_3, 3, 3, 1, 1, 512, "conv4_4")
    pool4 = add_maxpool_layer(conv4_4, 2, 2, 2, 2, "pool4")

    """block 5"""
    conv5_1 = add_conv_layer(pool4, 3, 3, 1, 1, 512, "conv5_1")
    conv5_2 = add_conv_layer(conv5_1, 3, 3, 1, 1, 512, "conv5_2")
    conv5_3 = add_conv_layer(conv5_2, 3, 3, 1, 1, 512, "conv5_3")
    conv5_4 = add_conv_layer(conv5_3, 3, 3, 1, 1, 512, "conv5_4")
    pool5 = add_maxpool_layer(conv5_4, 2, 2, 2, 2, "pool5")

    """block 6"""
    flat = tf.reshape(pool5, [-1, 7*7*512])
    embeddings = add_fc_layer(flat, 7*7*512, embedding_dim, "embedding", tf.nn.relu)
    #drop1 = add_dropout_layer(fc6, embedding_dim, "drop1")

    # fc7 = add_fc_layer(drop1, 4096, 4096, "fc7", tf.nn.relu)
    # drop2 = add_dropout_layer(fc7, , "drop2")

    # self.fc8 = add_fc_layer(drop2, 4096, self.NUMCLASSES, "fc8", tf.nn.relu)

    return embeddings 


