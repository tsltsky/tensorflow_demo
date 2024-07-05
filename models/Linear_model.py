from base.base_model import BaseModel
import tensorflow as tf

class LinearModel(BaseModel):
    def __init__(self, config):
        super(LinearModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def rmse(self, y_hat, y):
        log_preds = tf.log(y_hat)
        log_labels = tf.log(y)
        loss = tf.losses.mean_squared_error(labels=log_labels, predictions=log_preds)
        rmse = tf.sqrt(2 * tf.reduce_mean(loss))
        return rmse

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, [None, 354])
        self.y = tf.placeholder(tf.float32, [None, 1])

        #d1 = tf.layers.dense(self.x, 256, activation=tf.nn.relu, name="dense1")
        self.d2 = tf.layers.dense(self.x, 1, name="dense2")

        with tf.name_scope("loss"):
            self.lossvalue = tf.reduce_mean(tf.square(self.y - self.d2))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.lossvalue,
                                                                                         global_step=self.global_step_tensor)
            self.msevalue = self.rmse(self.y,self.d2)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)