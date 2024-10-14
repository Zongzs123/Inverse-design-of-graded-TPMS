import tensorflow as tf
from keras.callbacks import *
import numpy as np
from utils import load_data
from modules import forward_network
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
import pandas as pd

for i in range(0, 6):
    def train(args):
        from utils import loss_mae
        data = load_data(args.dataset_path)  # data_x represents structural parameters and data_y represents curve features
        data_x = data[:, :9]
        data_y = data[:, 9:]
        scaler1 = StandardScaler()
        scaler1.fit(data_x)
        data_x_standarized = scaler1.transform(data_x)

        scaler2 = StandardScaler()
        scaler2.fit(data_y)
        data_y_standarized = scaler2.transform(data_y)

        inputs = data_x_standarized
        targets = data_y_standarized

        # Define a learning rate scheduler callback
        lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, min_lr=0.000001, verbose=1)

        print('------------------------------------------------------------------------')
        print(f'Training ...')

        loss = []
        val_loss = []
        loss_MAE = []
        val_loss_MAE = []

        # train the model using the training data
        forward_model = forward_network()
        forward_model.summary()

        # Train again the model to obtain the history
        train_result = forward_model.fit(inputs, targets, epochs=args.epochs, batch_size=args.batch_size,
                                         callbacks=[lr_scheduler], verbose=2, validation_split=0.2, shuffle=True)

        loss.append(train_result.history['loss'])
        val_loss.append(train_result.history['val_loss'])
        loss_MAE.append(train_result.history['loss_mae'])
        val_loss_MAE.append(train_result.history['val_loss_mae'])

        loss = np.array(loss)
        val_loss = np.array(val_loss)
        loss_MAE = np.array(loss_MAE)
        val_loss_MAE = np.array(val_loss_MAE)

        loss_df = pd.DataFrame(loss.T)
        val_loss_df = pd.DataFrame(val_loss.T)
        loss_MAE_df = pd.DataFrame(loss_MAE.T)
        val_loss_MAE_df = pd.DataFrame(val_loss_MAE.T)

        # save the model of best hyperparameters
        base_dir = args.run_name
        os.makedirs(base_dir, exist_ok=True)
        model_loc = os.path.join(base_dir, 'forward'+str(i)+'.keras')
        forward_model.save(model_loc)

        # Save cv_results_ to a CSV file
        cv_results_file = os.path.join(base_dir, 'loss'+str(i)+'.xlsx')
        loss_df.to_excel(cv_results_file, index=False)
        cv_results_file = os.path.join(base_dir, 'val_loss'+str(i)+'.xlsx')
        val_loss_df.to_excel(cv_results_file, index=False)
        cv_results_file = os.path.join(base_dir, 'mae'+str(i)+'.xlsx')
        loss_MAE_df.to_excel(cv_results_file, index=False)
        cv_results_file = os.path.join(base_dir, 'val_mae'+str(i)+'.xlsx')
        val_loss_MAE_df.to_excel(cv_results_file, index=False)

        # printing the loss_MAE and val_loss_MAE results
        logging_info = "loss_batch_size" + str(args.batch_size) + "_mean:\n"
        print(logging_info, np.mean(loss[:, args.epochs - 1]))
        logging_info = "val_loss_batch_size" + str(args.batch_size) + "_mean:\n"
        print(logging_info, np.mean(val_loss[:, args.epochs - 1]))
        logging_info = "loss_MAE_batch_size" + str(args.batch_size) + "_mean:\n"
        print(logging_info, np.mean(loss_MAE[:, args.epochs - 1]))
        logging_info = "val_loss_MAE_batch_size" + str(args.batch_size) + "_mean:\n"
        print(logging_info, np.mean(val_loss_MAE[:, args.epochs - 1]))


    if __name__ == '__main__':
        import argparse
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.dataset_path = "train.xlsx"
        args.epochs = 300
        args.batch_size = 256
        args.run_name = "./forward_network_mlp/"
        train(args)
