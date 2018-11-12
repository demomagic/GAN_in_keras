import argparse
import importlib
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type = str, default = 'mnist')
    parser.add_argument('--model', type = str, default = 'wgan')
    args = parser.parse_args()
    model_type = args.model
    train_data = args.data
    datasets = importlib.import_module('keras.datasets.' + train_data)
    model = importlib.import_module('model.' + model_type)
    print('model type: %s, datasets: %s'%(model_type, train_data))
    
    save_path = 'gan_images/' + model_type
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    gan = model.GAN(datasets, save_path)
    gan.train_network(epochs = 100, batch_size = 64)