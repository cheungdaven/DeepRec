

class BaseRecommender:

    def __init__(self,  sess, learning_rate = 0.1, epoch = 500, N = 200, batch_size = 1024):
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def fit(self, train_data):
        self.train_data = train_data

    def save(self, save_path):
        self.save_path = save_path

    def evaluate(self, test_data, metrics):
        # RMSE
        if "RMSE" in metrics:
            error = 0
            test_set = list(test_data.keys())
            for (u, i) in test_set:
                pred_rating_test = \
                    self.sess.run([self.pred_y], feed_dict={self.cf_user_input: [u], self.cf_item_input: [i]})[0]
                if pred_rating_test < 0:
                    pred_rating_test = 0
                elif pred_rating_test > 5:
                    pred_rating_test = 5

                error += (float(test_data.get((u, i))) - pred_rating_test) ** 2

            print("RMSE:" + str(np.sqrt(error / len(test_set))[0]))

        # MAE

        # Precision

        # Recall

        #
        self.test_data = test_data
        print(self.test_data)

    def predict(self, user, item):
        self.user = user
        self.item = item



