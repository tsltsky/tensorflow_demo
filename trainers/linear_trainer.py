from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import pandas as pd

class LinearTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(LinearTrainer, self).__init__(sess, model, data, config,logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        mses = []
        for _ in loop:
            loss, rmse = self.train_step()
            losses.append(loss)
            mses.append(rmse)
        loss = np.mean(losses)
        mse = np.mean(mses)
        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'mse': mse,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss, acc, logits= self.sess.run([self.model.train_step, self.model.lossvalue, self.model.msevalue,self.model.d2],
                                     feed_dict=feed_dict)

        return loss, acc

    def eval(self,data):
        feed_dict = {self.model.x:data.eval_data()}
        logits = self.sess.run(self.model.d2,feed_dict=feed_dict)
        data.test_data["SalePrice"] = pd.Series(logits.reshape(1, -1)[0])
        submission = pd.concat([data.test_data['Id'],data.test_data['SalePrice']], axis=1)
        submission.to_csv('./submission.csv', index=False)




