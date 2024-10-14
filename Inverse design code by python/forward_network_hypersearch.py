import tensorflow as tf
from keras.callbacks import *
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
from utils import load_data
from modules import forward_network

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import make_scorer
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd


def train(args):
    from utils import loss_mae
    from utils import loss_function
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

    # Define the hyperparameter search space
    # Note that randomized search and grid search should use different dict !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    param_dist = {
        'units_layer1': [256],
        'units_layer2': [1600],
        'units_layer3': [1600],
        'units_layer4': [256],
        'optimizer': ['rmsprop'],
    }

    # Create the forward_model
    forward_model = KerasRegressor(build_fn=forward_network, epochs=args.epochs, batch_size=args.batch_size, verbose=2)

    # Create a scorer for the RandomizedSearchCV based on custom loss_MAE
    scorer = make_scorer(loss_function, greater_is_better=False)

    # Define a learning rate scheduler callback
    lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, min_lr=0.000001, verbose=1)

    if args.flag == 1:
        # Create the RandomizedSearchCV object with the callbacks
        random_search = RandomizedSearchCV(forward_model, param_distributions=param_dist, n_iter=50, cv=3, scoring=scorer,
                                           verbose=2)
    else:
        # Create the GridSearchCV object with the callbacks
        random_search = GridSearchCV(forward_model, param_dist, cv=2, scoring=scorer, verbose=2)

    print('------------------------------------------------------------------------')
    print(f'Training ...')

    # Fit the RandomizedSearchCV object to the data
    random_search.fit(inputs, targets, callbacks=[lr_scheduler])

    # Print the best hyperparameters
    print("Best Hyperparameters: ", random_search.best_params_)
    best_hyperparameters = random_search.best_params_
    print("all the result of CV: ", random_search.cv_results_)

    # Create a new model with the best hyperparameters
    best_forward_model = forward_network(units_layer1=best_hyperparameters['units_layer1'],
                                         units_layer2=best_hyperparameters['units_layer2'],
                                         units_layer3=best_hyperparameters['units_layer3'],
                                         units_layer4=best_hyperparameters['units_layer4'],
                                         optimizer=best_hyperparameters['optimizer'])

    # Train again the model to obtain the history
    train_result = best_forward_model.fit(inputs, targets, epochs=args.epochs, batch_size=args.batch_size,
                                          callbacks=[lr_scheduler], verbose=2, validation_split=0.2)

    # save the model of best hyperparameters
    base_dir = args.run_name
    os.makedirs(base_dir, exist_ok=True)
    model_loc = os.path.join(base_dir, 'forward_model_4ceng_res_final_4.keras')
    best_forward_model.save(model_loc)

    # convert cv_results_ to dataframe
    cv_results_df = pd.DataFrame(random_search.cv_results_)

    # Save cv_results_ to a CSV file
    cv_results_file = os.path.join(base_dir, 'cv_results_4ceng_res_final_4.csv')
    cv_results_df.to_csv(cv_results_file, index=False)

    # Print all the results of CV
    print("All the results of CV:\n", cv_results_df)

    # # plot the figure of training result
    # loss_MAE = train_result.history['loss_mae']
    # val_loss_MAE = train_result.history['val_loss_mae']
    # loss = train_result.history['loss']
    # val_loss = train_result.history['val_loss']
    # epochs = range(len(loss_MAE))
    #
    # plt.plot(epochs, loss_MAE, 'b', label='Training absolute mean error')
    # plt.plot(epochs, val_loss_MAE, 'r', label='Validation absolute mean error')
    # plt.title('Training and validation absolute mean error')
    # plt.legend()
    # plt.figure()
    #
    # plt.plot(epochs, loss, 'b', label='Training loss')
    # plt.plot(epochs, val_loss, 'r', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.legend()
    # plt.figure()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "./forward_network_gird/"
    args.epochs = 300
    args.batch_size = 64
    args.dataset_path = "train.xlsx"
    args.flag = 0  # 1 for random search, 0 for grid search
    train(args)
