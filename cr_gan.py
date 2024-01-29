from keras.layers import Input, Embedding, Dense, concatenate, Flatten, Subtract
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam
from utils import *
from time import time

ranker_layers = [128, 64, 32, 8]
ranker_reg_layers = [0 for _ in range(len(ranker_layers))]
discriminator_layers = [32, 8]
discriminator_reg_layers = [0 for _ in range(len(discriminator_layers))]
K = 10
epochs = 15
d_lr, r_lr = 0.01, 0.05

out = True
dataset = 'ml100k'
now = time()
ranker_out_file = 'checkpoints/%d_%s_Ranker_%s.h5' % (now, dataset, ranker_layers)
dis_out_file = 'checkpoints/%d_%s_Dis_%s.h5' % (now, dataset, discriminator_layers)
log_file = 'checkpoints/%d_%s_%d.txt' % (now, dataset, K)

resume = False
in_time = 1549967959
in_path = 'checkpoints'
ranker_in_file = '%s/%s_%s_Ranker_%s.h5' % (in_path, in_time, dataset, ranker_layers)
dis_in_file = '%s/%s_%s_Dis_%s.h5' % (in_path, in_time, dataset, discriminator_layers)

num_users, num_items, train_u_input, train_i_input, train_j_input = get_pairwise_train_dataset(
    path='data/%s_train.dat' % dataset)
print(num_users, num_items)
# num_users, num_items = 943, 1683
# num_users, num_items = 9649, 17770
num_users += 1
num_items += 1
train_labels = [1 for _ in range(len(train_u_input))]
print(len(train_u_input))

testItems, testRatings = get_test_data(path='data/%s_test_ratings.lsvm' % dataset)


input = Input((1,), dtype='float32', name='input')
embedding_u = Embedding(input_dim=num_users, output_dim=int((discriminator_layers[0] - 1) / 3),
                        name='rank_embedding_item',
                        init='random_normal', W_regularizer=l2(discriminator_reg_layers[0]), input_length=1)
latent_u = Flatten()(embedding_u(input))
embedding_model_u = Model(input=input, output=latent_u)
train_u_latent = embedding_model_u.predict(train_u_input)

embedding_i = Embedding(input_dim=num_items, output_dim=int((discriminator_layers[0] - 1) / 3),
                        name='rank_embedding_item',
                        init='random_normal', W_regularizer=l2(discriminator_reg_layers[0]), input_length=1)
latent_i = Flatten()(embedding_i(input))
embedding_model_i = Model(input=input, output=latent_i)
train_i_latent = embedding_model_i.predict(train_i_input)
train_j_latent = embedding_model_i.predict(train_j_input)


class CRGAN:

    def build_ranker(self):
        r_u_input = Input(shape=(1,), dtype='int32', name='user_input')
        r_i_input = Input(shape=(1,), dtype='int32', name='item_input')

        Rank_Embedding_User = Embedding(input_dim=num_users, output_dim=int(ranker_layers[0] / 2),
                                        name='rank_embedding_user',
                                        init='random_normal', W_regularizer=l2(ranker_reg_layers[0]), input_length=1)
        Rank_Embedding_Item = Embedding(input_dim=num_items, output_dim=int(ranker_layers[0] / 2),
                                        name='rank_embedding_item',
                                        init='random_normal', W_regularizer=l2(ranker_reg_layers[0]), input_length=1)

        r_u_latent = Flatten()(Rank_Embedding_User(r_u_input))
        r_i_latent = Flatten()(Rank_Embedding_Item(r_i_input))
        vector = concatenate([r_u_latent, r_i_latent], axis=-1)
        for idx in range(1, len(ranker_layers)):
            layer = Dense(ranker_layers[idx], W_regularizer=l2(ranker_reg_layers[idx]), activation='relu',
                          name='r_layer%d' % idx)
            vector = layer(vector)
        prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='r_prediction')(vector)
        ranker = Model(input=[r_u_input, r_i_input],
                       output=prediction)
        return ranker

    def build_discriminator(self):
        d_u_input = Input((int((discriminator_layers[0] - 1) / 3),), dtype='float32', name='d_u_input')
        d_i_input = Input((int((discriminator_layers[0] - 1) / 3),), dtype='float32', name='d_i_input')
        d_j_input = Input((int((discriminator_layers[0] - 1) / 3),), dtype='float32', name='d_j_input')
        d_r_input = Input((1,), dtype='float32', name='d_r_input')
        d_input = concatenate([d_u_input, d_i_input, d_j_input, d_r_input], axis=-1)
        vector = Dense(discriminator_layers[0], W_regularizer=l2(discriminator_reg_layers[0]), activation='relu',
                       name='d_layer0')(d_input)
        for idx in range(1, len(discriminator_layers)):
            vector = Dense(discriminator_layers[idx], W_regularizer=l2(discriminator_reg_layers[idx]),
                           activation='relu', name='d_layer%d' % idx)(vector)
        prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='d_prediction')(vector)
        discriminator = Model(input=[d_u_input, d_i_input, d_j_input, d_r_input], output=prediction)
        return discriminator

    def __init__(self, d_lr=d_lr, r_lr=r_lr):
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.summary()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=Adam(d_lr),
                                   metrics=['accuracy'])

        # Build the generator
        u_input = Input(shape=(1,), dtype='float32', name='u_input')
        i_input = Input(shape=(1,), dtype='float32', name='i_input')
        j_input = Input(shape=(1,), dtype='float32', name='j_input')
        Dis_Embedding_U = Embedding(input_dim=num_users, output_dim=int((discriminator_layers[0] - 1) / 3),
                                    name='dis_embedding_u',
                                    init='random_normal', W_regularizer=l2(discriminator_reg_layers[0]), input_length=1)
        Dis_Embedding_I = Embedding(input_dim=num_items, output_dim=int((discriminator_layers[0] - 1) / 3),
                                    name='dis_embedding_i',
                                    init='random_normal', W_regularizer=l2(discriminator_reg_layers[0]), input_length=1)
        Dis_Embedding_J = Embedding(input_dim=num_items, output_dim=int((discriminator_layers[0] - 1) / 3),
                                    name='dis_embedding_j',
                                    init='random_normal', W_regularizer=l2(discriminator_reg_layers[0]), input_length=1)

        d_u_latent = Flatten()(Dis_Embedding_U(u_input))
        d_i_latent = Flatten()(Dis_Embedding_I(i_input))
        d_j_latent = Flatten()(Dis_Embedding_J(j_input))
        self.ranker = self.build_ranker()
        r_i = self.ranker([u_input, i_input])
        r_j = self.ranker([u_input, j_input])
        r = Subtract()([r_i, r_j])
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        y_pred = self.discriminator([d_u_latent, d_i_latent, d_j_latent, r])
        self.combined = Model([u_input, i_input, j_input], y_pred)
        self.combined.summary()
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=Adam(r_lr),
                              metrics=['accuracy'])

    def train(self):

        if resume is True:
            self.ranker.load_weights(ranker_in_file)
            # self.discriminator.load_weights(dis_in_file)
        metrics = evaluate_model(self.ranker, testItems, testRatings, K)

        print('init: ', metrics)
        with open(log_file, 'w') as log:
            print('-1', ' '.join('%.4f' % i for i in metrics), file=log)
        best_metrics, best_epoch = metrics, -1
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            gen_r_i = self.ranker.predict([np.array(train_u_input), np.array(train_i_input)])
            gen_r_j = self.ranker.predict([np.array(train_u_input), np.array(train_j_input)])
            gen_r = gen_r_i - gen_r_j
            # print(gen_r_j)

            valid = np.ones_like(gen_r)
            d_loss_real = self.discriminator.train_on_batch(
                [train_u_latent, train_i_latent, train_j_latent, np.array(train_labels)],
                valid)
            fake = np.zeros_like(gen_r)
            d_loss_gen = self.discriminator.train_on_batch(
                [train_u_latent, train_i_latent, train_j_latent, gen_r], fake)
            d_loss = (d_loss_real[0] + d_loss_gen[0]) / 2  # 0: loss, 1: acc
            # print('epoch %d : d_loss = %.4f' % (epoch, d_loss))

            # ---------------------
            #  Train Generator
            # ---------------------
            g_loss = self.combined.train_on_batch([train_u_input, train_i_input, train_j_input], valid)
            metrics = evaluate_model(self.ranker, testItems, testRatings, K)
            with open(log_file, 'a') as log:
                important_index = 1
                if metrics[important_index] > best_metrics[important_index]:
                    best_metrics, best_epoch = metrics, epoch
                    print('epoch %d: ' % epoch, metrics, '[best]')
                    print('%d' % epoch, ' '.join('%.4f' % i for i in metrics), g_loss[0], '[best]', file=log)
                    if out is True:
                        self.ranker.save_weights(ranker_out_file, overwrite=True)
                        self.discriminator.save_weights(dis_out_file, overwrite=True)
                else:
                    print('epoch %d: ' % epoch, metrics)
                    print('%d' % epoch, ' '.join('%.4f' % i for i in metrics), g_loss[0], file=log)


cr_gan = CRGAN()
cr_gan.train()
