import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from keras.models import Model,Sequential
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import Input, Dense, Flatten, Reshape, LeakyReLU

class GAN():
    def __init__(self, datasets, save_path, latent_dim = 100, learning_rate = 0.00005):
        (self.X_train, _), (_, _) = datasets.load_data()
        (_, self.image_width, self.image_height) = self.X_train.shape
        self.channel = 1
        self.input_shape = (self.image_width, self.image_height, self.channel)
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.save_path = save_path
        
        self.gen = self.generator()
        self.dis = self.discriminator()
        
        #only train the generator
        gen_input = Input(shape = (self.latent_dim,))
        imgs = self.gen(gen_input)
        dis_output = self.dis(imgs)
        self.gan_dis = Model(gen_input, dis_output)
        opt = RMSprop(lr = self.learning_rate)
        self.gan_dis.compile(optimizer = opt, loss = self.wasserstein_loss, metrics = ['accuracy'])

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
        model.add(Conv2D(64, (5, 5), padding='same', input_shape=(self.image_width, self.image_height, self.channel)))
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

        model = Model(image, outputs)
        opt = RMSprop(lr = self.learning_rate)
        model.compile(optimizer = opt, loss = self.wasserstein_loss, metrics = ['accuracy'])
        
        return model
    
    def train_network(self, epochs = 100, batch_size = 64):
        d_iters = 5
        batch_num = len(self.X_train) // batch_size
        # Rescale -1 to 1
        X_train = (self.X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        
        for epoch in range(epochs):
            for _ in range(batch_num):
                self.dis.trainable = True
                for _ in range(d_iters):
                    images = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
                    noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                    fake_images = self.gen.predict(noise)
                    # Train the discriminator
                    d_loss_real = self.dis.train_on_batch(images, valid)
                    d_loss_fake = self.dis.train_on_batch(fake_images, fake)
                    d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                    
                    self.clip_d_weights()
                
                self.dis.trainable = False
                g_loss = self.gan_dis.train_on_batch(noise, valid)
                # Plot the progress
                print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))
            
            self.sample_images(epoch)
    
    def clip_d_weights(self):
        weights = [np.clip(w, -0.01, 0.01) for w in self.dis.get_weights()]
        self.dis.set_weights(weights)
        
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
    
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
