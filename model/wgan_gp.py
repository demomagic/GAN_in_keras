import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from functools import partial
from keras.optimizers import Adam
from keras.layers.merge import _Merge
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import Input, Dense, Flatten, Reshape, LeakyReLU

class GAN():
    def __init__(self, datasets, save_path, latent_dim = 100, learning_rate = 0.0001):
        (self.X_train, _), (_, _) = datasets.load_data()
        (_, self.image_width, self.image_height) = self.X_train.shape
        self.channel = 1
        self.input_shape = (self.image_width, self.image_height, self.channel)
        self.latent_dim = latent_dim
        
        self.learning_rate = learning_rate
        self.save_path = save_path
        
        self.gen = self.generator()
        self.dis = self.discriminator()
                
    def generator(self):
        noise = Input(shape = (self.latent_dim,))
        
        model = Sequential()
        model.add(Dense(1024, input_dim = self.latent_dim))
        model.add(LeakyReLU())
        model.add(Dense(128 * 7 * 7))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
        model.add(Conv2DTranspose(128, (5, 5), strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2D(64, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2D(self.channel, (5, 5), padding='same', activation='tanh'))
        
        image = model(noise)
        model.summary()
        
        return Model(noise, image)
        
    def discriminator(self):
        image = Input(shape = self.input_shape)
        
        model = Sequential()
        model.add(Conv2D(64, (5, 5), padding='same', input_shape=self.input_shape))
        model.add(LeakyReLU())
        model.add(Conv2D(128, (5, 5), kernel_initializer='he_normal', strides=[2, 2]))
        model.add(LeakyReLU())
        model.add(Conv2D(128, (5, 5), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
        model.add(LeakyReLU())
        model.add(Flatten())
        model.add(Dense(1024, kernel_initializer='he_normal'))
        model.add(LeakyReLU())
        model.add(Dense(1, kernel_initializer='he_normal'))
        
        outputs = model(image)
        model.summary()
        
        return Model(image, outputs)

    def build_training_method(self):
        opt = Adam(lr = self.learning_rate,  beta_1=0., beta_2=0.9)
        
        self.gen.trainable = False
        real_image = Input(shape = self.input_shape)
        dis_random_input = Input(shape=(self.latent_dim,))
        fake_image = self.gen(dis_random_input)
        dis_fake = self.dis(fake_image)
        dis_real = self.dis(real_image)
        averaged_samples = RandomWeightedAverage()([real_image, fake_image])
        averaged_samples_out = self.dis(averaged_samples)
        partial_gp_loss = partial(self.gradient_penalty_loss, averaged_samples=averaged_samples)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names
        self.dis_model = Model(inputs = [real_image, dis_random_input], outputs = [dis_real, dis_fake, averaged_samples_out])
        self.dis_model.compile(loss = [self.wasserstein_loss, self.wasserstein_loss, partial_gp_loss], optimizer = opt, loss_weights = [1, 1, 10])
        
        self.gen.trainable = True
        self.dis.trainable = False
        gen_random_input = Input(shape=(self.latent_dim,))
        fake_image = self.gen(gen_random_input)
        dis_fake = self.dis(fake_image)
        self.gen_model = Model(gen_random_input, dis_fake)
        self.gen_model.compile(loss = self.wasserstein_loss, optimizer = opt)
        
    def train_network(self, epochs = 100, batch_size = 64):
        self.build_training_method()
        
        d_iters = 5
        batch_num = len(self.X_train) // batch_size

        # Rescale -1 to 1
        X_train = (self.X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # for gradient penalty
        
        for epoch in range(epochs):
            for _ in range(batch_num):
                # Train the discriminator
                self.gen.trainable = False
                self.dis.trainable = True
                for _ in range(d_iters):
                    images = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
                    noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                    d_loss = self.dis_model.train_on_batch([images, noise], [valid, fake, dummy])
                
                # Train the generator  
                self.gen.trainable = True
                self.dis.trainable = False
                g_loss = self.gen_model.train_on_batch(noise, valid)
                
            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
                
            self.sample_images(epoch)
    
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        gradients = K.gradients(K.sum(y_pred), averaged_samples)
        gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
        gradient_penalty = K.square(1 - gradient_l2_norm)
        return gradient_penalty
    
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.gen.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 1

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(self.save_path + "/%d.png" % epoch)
        plt.close()
        
class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((64, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])