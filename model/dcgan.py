import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model,Sequential
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers import Input, Dense, Flatten, Reshape, Activation, UpSampling2D, MaxPooling2D

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
        
        #only train the generator
        self.dis.trainable = False
        gen_input = Input(shape = (self.latent_dim,))
        imgs = self.gen(gen_input)
        dis_output = self.dis(imgs)
        self.gan_dis = Model(gen_input, dis_output)
        opt = SGD(lr=self.learning_rate, momentum=0.9, nesterov=True)
        self.gan_dis.compile(optimizer = opt, loss = 'binary_crossentropy')
        
    def generator(self):
        noise = Input(shape = (self.latent_dim,))
        
        model = Sequential()
        model.add(Dense(input_dim=self.latent_dim, units=1024))
        model.add(Activation('tanh'))
        model.add(Dense(128*7*7))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(64, (5, 5), padding='same'))
        model.add(Activation('tanh'))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(1, (5, 5), padding='same'))
        model.add(Activation('tanh'))
        
        image = model(noise)
        
        return Model(noise, image)
    
    def discriminator(self):
        image = Input(shape = self.input_shape)
        
        model = Sequential()
        model.add(Conv2D(64, (5, 5), padding='same', input_shape=self.input_shape))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (5, 5)))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('tanh'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        
        output = model(image)
        model = Model(image, output)
        opt = SGD(lr=self.learning_rate, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        
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
                                    
                # Train the generator
                self.dis.trainable = False
                g_loss = self.gan_dis.train_on_batch(noise, valid)
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
