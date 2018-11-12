import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import Input, Dense, Flatten, Reshape, LeakyReLU

class GAN():
    def __init__(self, datasets, save_path, latent_dim = 100, learning_rate = 0.0005):
        (self.X_train, _), (_, _) = datasets.load_data()
        (_, self.image_width, self.image_height) = self.X_train.shape
        self.channel = 1
        self.input_shape = (self.image_width, self.image_height, self.channel)
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.save_path = save_path
        
        self.gen = self.generator()
        self.dis = self.discriminator()

        self.real_image, self.random_input, self.fake_image, self.d_loss, self.g_loss, self.d_opt, self.g_opt = self.build_training_method()
        
        self.sess = tf.Session()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

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
        real_image = tf.placeholder(tf.float32, shape=[None, self.image_width, self.image_height, self.channel])
        random_input = tf.placeholder(tf.float32, shape=[None, self.latent_dim])
        
        fake_image = self.gen(random_input)
        real_result = self.dis(real_image)
        
        fake_result = self.dis(fake_image)
        d_loss = -(tf.reduce_sum(real_result) - tf.reduce_sum(fake_result))
        g_loss = -tf.reduce_sum(fake_result)
        
        # gradient penalty
        alpha = tf.random_uniform((tf.shape(real_image)[0], 1, 1, 1), minval = 0., maxval = 1,)
        differ = fake_image - real_image
        interp = real_image + (alpha * differ)
        grads = tf.gradients(self.dis(interp), [interp])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices = [3]))
        grad_penalty = tf.reduce_mean((slopes - 1.)**2)
        d_loss += 10 * grad_penalty
        
        d_opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1 = 0., beta2 = 0.9)\
                    .minimize(d_loss, var_list = self.dis.trainable_weights)
        g_opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1 = 0., beta2 = 0.9)\
                    .minimize(g_loss, var_list = self.gen.trainable_weights)
        
        return real_image, random_input, fake_image, d_loss, g_loss, d_opt, g_opt
        
    def train_network(self, epochs = 100, batch_size = 64):
        
        d_iters = 5
        batch_num = len(self.X_train) // batch_size

        # Rescale -1 to 1
        X_train = (self.X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        
        for epoch in range(epochs):
            for _ in range(batch_num):
                for _ in range(d_iters):
                    images = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
                    noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                    # Train the discriminator
                    _, d_loss = self.sess.run([self.d_opt, self.d_loss], feed_dict={self.real_image: images, self.random_input: noise, K.learning_phase(): 1})
                
                # Train the generator
                _, g_loss = self.sess.run([self.g_opt, self.g_loss], feed_dict = {self.random_input: noise, K.learning_phase(): 1})
                
                # Plot the progress
                print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss, g_loss))
                
            self.sample_images(epoch)
    
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
