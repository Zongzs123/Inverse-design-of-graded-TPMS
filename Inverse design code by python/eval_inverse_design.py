import tensorflow as tf
import numpy as np
from utils import load_data
from modules import forward_network, inverse_network

from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from keras import backend as K
import pandas as pd


def main(args):
    train_data = load_data(args.train_path)
    test_data = load_data(args.test_path)
    # columns = ["V1a", "V1v", "V1c", "w", "meshVolume", "relativeVolume", "surfaceArea", "relativeArea", "thickness",
    #            "poreDiameter", "thicknessAM", "areaMean", "areaMin", "areaMax", "s1", "s2", "s3", "s4", "s5", "s6",
    #            "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20"]
    # columns = ["V1a", "V1v", "V1c", "w", "relativeVolume", "relativeArea", "thickness", "poreDiameter", "areaMean",
    #            "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16",
    #            "s17", "s18", "s19", "s20"]
    columns = ["V1a", "V1v", "V1c", "w", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12",
               "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20"]
    tmp_data_x = train_data[:, :4]
    tmp_data_y = train_data[:, 4:]
    data_x = test_data[:, :4]
    data_y = test_data[:, 4:]

    # transform the test data into the format of training data
    scaler1 = StandardScaler()
    scaler1.fit(tmp_data_x)
    scaler2 = StandardScaler()
    scaler2.fit(tmp_data_y)
    data_x_standarized = scaler1.transform(data_x)
    data_y_standarized = scaler2.transform(data_y)

    test_data_x = data_x_standarized
    test_data_y = data_y_standarized

    inverse_model_1 = inverse_network()
    inverse_model_2 = inverse_network()
    inverse_model_3 = inverse_network()
    inverse_model_4 = inverse_network()
    inverse_model_5 = inverse_network()
    inverse_model_6 = inverse_network()
    forward_model_1 = forward_network()
    forward_model_2 = forward_network()
    forward_model_3 = forward_network()
    forward_model_4 = forward_network()
    forward_model_5 = forward_network()
    forward_model_6 = forward_network()

    # specify the inverse network 1 and obtain the result of inverse network 1
    inverse_model_1.load_weights(args.inverse_model_loc_1, by_name=False)
    inverse_model_1.trainable = False
    predicted_x_before_transform_1 = inverse_model_1(test_data_y, training=False)

    # specify the inverse network 2 and obtain the result of inverse network 2
    inverse_model_2.load_weights(args.inverse_model_loc_2, by_name=False)
    inverse_model_2.trainable = False
    predicted_x_before_transform_2 = inverse_model_2(test_data_y, training=False)

    # specify the inverse network 3 and obtain the result of inverse network 3
    inverse_model_3.load_weights(args.inverse_model_loc_3, by_name=False)
    inverse_model_3.trainable = False
    predicted_x_before_transform_3 = inverse_model_3(test_data_y, training=False)

    # specify the inverse network 4 and obtain the result of inverse network 4
    inverse_model_4.load_weights(args.inverse_model_loc_4, by_name=False)
    inverse_model_4.trainable = False
    predicted_x_before_transform_4 = inverse_model_4(test_data_y, training=False)

    # specify the inverse network 5 and obtain the result of inverse network 5
    inverse_model_5.load_weights(args.inverse_model_loc_5, by_name=False)
    inverse_model_5.trainable = False
    predicted_x_before_transform_5 = inverse_model_5(test_data_y, training=False)

    # specify the inverse network 6 and obtain the result of inverse network 6
    inverse_model_6.load_weights(args.inverse_model_loc_6, by_name=False)
    inverse_model_6.trainable = False
    predicted_x_before_transform_6 = inverse_model_6(test_data_y, training=False)

    # specify the forward model 1 and obtain the result of forward model 1
    forward_model_1.load_weights(args.forward_model_loc_1, by_name=False)
    forward_model_1.trainable = False
    predicted_y_before_transform_1 = forward_model_1(predicted_x_before_transform_1, training=False)

    # specify the forward model 2 and obtain the result of forward model 2
    forward_model_2.load_weights(args.forward_model_loc_2, by_name=False)
    forward_model_2.trainable = False
    predicted_y_before_transform_2 = forward_model_2(predicted_x_before_transform_2, training=False)

    # specify the forward model 3 and obtain the result of forward model 3
    forward_model_3.load_weights(args.forward_model_loc_3, by_name=False)
    forward_model_3.trainable = False
    predicted_y_before_transform_3 = forward_model_3(predicted_x_before_transform_3, training=False)

    # specify the forward model 4 and obtain the result of forward model 4
    forward_model_4.load_weights(args.forward_model_loc_4, by_name=False)
    forward_model_4.trainable = False
    predicted_y_before_transform_4 = forward_model_4(predicted_x_before_transform_4, training=False)

    # specify the forward model 5 and obtain the result of forward model 5
    forward_model_5.load_weights(args.forward_model_loc_5, by_name=False)
    forward_model_5.trainable = False
    predicted_y_before_transform_5 = forward_model_5(predicted_x_before_transform_5, training=False)

    # specify the forward model 6 and obtain the result of forward model 6
    forward_model_6.load_weights(args.forward_model_loc_6, by_name=False)
    forward_model_6.trainable = False
    predicted_y_before_transform_6 = forward_model_6(predicted_x_before_transform_6, training=False)

    # obtain the predicted data using inverse network 1
    predicted_x_points_1 = scaler1.inverse_transform(predicted_x_before_transform_1)
    predicted_y_points_1 = scaler2.inverse_transform(predicted_y_before_transform_1)

    # obtain the predicted data using inverse network 2
    predicted_x_points_2 = scaler1.inverse_transform(predicted_x_before_transform_2)
    predicted_y_points_2 = scaler2.inverse_transform(predicted_y_before_transform_2)

    # obtain the predicted data using inverse network 3
    predicted_x_points_3 = scaler1.inverse_transform(predicted_x_before_transform_3)
    predicted_y_points_3 = scaler2.inverse_transform(predicted_y_before_transform_3)

    # obtain the predicted data using inverse network 4
    predicted_x_points_4 = scaler1.inverse_transform(predicted_x_before_transform_4)
    predicted_y_points_4 = scaler2.inverse_transform(predicted_y_before_transform_4)

    # obtain the predicted data using inverse network 5
    predicted_x_points_5 = scaler1.inverse_transform(predicted_x_before_transform_5)
    predicted_y_points_5 = scaler2.inverse_transform(predicted_y_before_transform_5)

    # obtain the predicted data using inverse network 6
    predicted_x_points_6 = scaler1.inverse_transform(predicted_x_before_transform_6)
    predicted_y_points_6 = scaler2.inverse_transform(predicted_y_before_transform_6)

    # select the best prediction and evaluate using our defined metrics
    error_ratio_array = []
    area_ground_truth = []
    predicted_x_points = []
    predicted_y_points = []
    for i in range(data_y.shape[0]):
        inverse_design_predicted_error_1 = K.sum(K.square(predicted_y_points_1[i, :] - data_y[i, :]))
        inverse_design_predicted_error_2 = K.sum(K.square(predicted_y_points_2[i, :] - data_y[i, :]))
        inverse_design_predicted_error_3 = K.sum(K.square(predicted_y_points_3[i, :] - data_y[i, :]))
        inverse_design_predicted_error_4 = K.sum(K.square(predicted_y_points_4[i, :] - data_y[i, :]))
        inverse_design_predicted_error_5 = K.sum(K.square(predicted_y_points_5[i, :] - data_y[i, :]))
        inverse_design_predicted_error_6 = K.sum(K.square(predicted_y_points_6[i, :] - data_y[i, :]))
        error_ratio_1 = K.sqrt(tf.cast(inverse_design_predicted_error_1, dtype=tf.float32) / 20) / (K.sum(data_y[i, :])/20)
        error_ratio_2 = K.sqrt(tf.cast(inverse_design_predicted_error_2, dtype=tf.float32) / 20) / (K.sum(data_y[i, :])/20)
        error_ratio_3 = K.sqrt(tf.cast(inverse_design_predicted_error_3, dtype=tf.float32) / 20) / (K.sum(data_y[i, :])/20)
        error_ratio_4 = K.sqrt(tf.cast(inverse_design_predicted_error_4, dtype=tf.float32) / 20) / (K.sum(data_y[i, :])/20)
        error_ratio_5 = K.sqrt(tf.cast(inverse_design_predicted_error_5, dtype=tf.float32) / 20) / (K.sum(data_y[i, :])/20)
        error_ratio_6 = K.sqrt(tf.cast(inverse_design_predicted_error_6, dtype=tf.float32) / 20) / (K.sum(data_y[i, :])/20)
        best_flag = np.argmin(
            [error_ratio_1, error_ratio_2, error_ratio_3, error_ratio_4, error_ratio_5, error_ratio_6])
        if best_flag == 0:
            error_ratio_array.append(error_ratio_1)
            predicted_x_points.append(predicted_x_points_1[i, :])
            predicted_y_points.append(predicted_y_points_1[i, :])
        elif best_flag == 1:
            error_ratio_array.append(error_ratio_2)
            predicted_x_points.append(predicted_x_points_2[i, :])
            predicted_y_points.append(predicted_y_points_2[i, :])
        elif best_flag == 2:
            error_ratio_array.append(error_ratio_3)
            predicted_x_points.append(predicted_x_points_3[i, :])
            predicted_y_points.append(predicted_y_points_3[i, :])
        elif best_flag == 3:
            error_ratio_array.append(error_ratio_4)
            predicted_x_points.append(predicted_x_points_4[i, :])
            predicted_y_points.append(predicted_y_points_4[i, :])
        elif best_flag == 4:
            error_ratio_array.append(error_ratio_5)
            predicted_x_points.append(predicted_x_points_5[i, :])
            predicted_y_points.append(predicted_y_points_5[i, :])
        else:
            error_ratio_array.append(error_ratio_6)
            predicted_x_points.append(predicted_x_points_6[i, :])
            predicted_y_points.append(predicted_y_points_6[i, :])

        area_ground_truth.append(K.sum(data_y[i, :])/20)

    error_ratio_df = pd.DataFrame(np.array(error_ratio_array))
    area_ground_truth_df = pd.DataFrame(np.array(area_ground_truth))
    error_ratio_df.to_excel('train_4para_nrmse.xlsx', index=False)  # the name can be changed for different purposes
    area_ground_truth_df.to_excel('train_4para_average.xlsx', index=False)  # the name can be changed for different purposes
    mean_error_ratio = tf.reduce_mean(error_ratio_array)
    print(len(error_ratio_array))
    print(mean_error_ratio)

    # predicted_x_points = np.array(predicted_x_points)
    # predicted_y_points = np.array(predicted_y_points)
    # predicted_points = np.concatenate((predicted_x_points, predicted_y_points), axis=-1)
    # print(predicted_points.shape)
    # predicted_points_df = pd.DataFrame(predicted_points)
    # predicted_points_df.to_excel('predicted_test.xlsx', index=False)  # the name can be changed for different purposes


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.train_path = "train-4canshu.xlsx"
    args.test_path = "train-4canshu.xlsx"
    args.inverse_model_loc_1 = "inverse0_4para.keras"
    args.inverse_model_loc_2 = "inverse1_4para.keras"
    args.inverse_model_loc_3 = "inverse2_4para.keras"
    args.inverse_model_loc_4 = "inverse3_4para.keras"
    args.inverse_model_loc_5 = "inverse4_4para.keras"
    args.inverse_model_loc_6 = "inverse5_4para.keras"
    args.forward_model_loc_1 = "forward0_4para.keras"
    args.forward_model_loc_2 = "forward1_4para.keras"
    args.forward_model_loc_3 = "forward2_4para.keras"
    args.forward_model_loc_4 = "forward3_4para.keras"
    args.forward_model_loc_5 = "forward4_4para.keras"
    args.forward_model_loc_6 = "forward5_4para.keras"
    main(args)
