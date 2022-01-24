import tensorflow as tf

nn_weights_init = tf.contrib.layers.variance_scaling_initializer()

def conv2D_32(X,
              bn_training,
              filter_size=(5,5),
              strides=(2, 2),
              n_filters=4,
              scope='conv'):

    with tf.variable_scope(scope+'h1'):

        h1 = tf.layers.conv2d(X, 
                              filters=n_filters,
                              kernel_size=filter_size, 
                              strides=strides,
                              padding='same',
                              activation=tf.nn.relu,
                              kernel_initializer=nn_weights_init)

        h1 = tf.layers.batch_normalization(h1,training=bn_training)

    with tf.variable_scope(scope+'h2'):

        h2 = tf.layers.conv2d(h1, 
                              filters=n_filters*4,
                              kernel_size=filter_size, 
                              strides=strides,
                              padding='same',
                              activation=tf.nn.relu,
                              kernel_initializer=nn_weights_init)

        h2 = tf.layers.batch_normalization(h2,training=bn_training)

    with tf.variable_scope(scope+'h3'):

        h3 = tf.layers.conv2d(h2, 
                              filters=n_filters*16,
                              kernel_size=filter_size, 
                              strides=strides,
                              padding='same',
                              activation=tf.nn.relu,
                              kernel_initializer=nn_weights_init)

        h3 = tf.layers.batch_normalization(h3,training=bn_training)

    return h3


def upconv2D_32(Z,
                bn_training,
                filter_size=(5,5),
                strides=(2, 2),
                n_filters=4,
                scope='upconv'):

    with tf.variable_scope(scope+'h1'):

        h1 = tf.layers.conv2d_transpose(Z,
                                        filters=n_filters*16,
                                        kernel_size=filter_size,
                                        strides=strides,
                                        padding='same', 
                                        activation=tf.nn.relu,
                                        kernel_initializer=nn_weights_init,
                                        use_bias=False)

        h1 = tf.layers.batch_normalization(h1,training=bn_training)

    with tf.variable_scope(scope+'h2'):

        h2 = tf.layers.conv2d_transpose(h1,
                                        filters =n_filters*4,
                                        kernel_size=filter_size,
                                        strides=strides,
                                        padding='same', 
                                        activation=tf.nn.relu,
                                        kernel_initializer=nn_weights_init,
                                        use_bias=False)

        h2 = tf.layers.batch_normalization(h2,training=bn_training)

    with tf.variable_scope(scope+'h3'):

        h3 = tf.layers.conv2d_transpose(h2,
                                        filters=n_filters,
                                        kernel_size=filter_size,
                                        strides=strides,
                                        padding='same', 
                                        activation=tf.nn.relu,
                                        kernel_initializer=nn_weights_init,
                                        use_bias=False)

        h3 = tf.layers.batch_normalization(h3,training=bn_training)

    return h3
