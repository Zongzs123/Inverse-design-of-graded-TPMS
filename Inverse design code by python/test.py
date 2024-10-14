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
    # columns = ["V1a", "V1v", "V1c", "w", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12",
    #            "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20"]
    # columns = ["V1a", "V1v", "V1c", "w", "meshVolume", "relativeVolume", "surfaceArea", "relativeArea", "thickness",
    #            "poreDiameter", "thicknessAM", "areaMean", "areaMin", "areaMax", "s1", "s2", "s3", "s4", "s5", "s6",
    #            "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20"]
    columns = ["V1a", "V1v", "V1c", "w", "relativeVolume", "relativeArea", "thickness", "poreDiameter", "areaMean",
               "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16",
               "s17", "s18", "s19", "s20"]
    tmp_data_x = train_data[:, :9]
    tmp_data_y = train_data[:, 9:]
    data_x = test_data[:, :9]
    data_y = test_data[:, 9:]

    # transform the test data into the format of training data
    scaler1 = StandardScaler()
    scaler1.fit(tmp_data_x)
    scaler2 = StandardScaler()
    scaler2.fit(tmp_data_y)
    data_x_standarized = scaler1.transform(data_x)
    data_y_standarized = scaler2.transform(data_y)

    test_data_x = data_x_standarized
    test_data_y = data_y_standarized

    forward_model_1 = forward_network()
    forward_model_2 = forward_network()
    forward_model_3 = forward_network()
    forward_model_4 = forward_network()
    forward_model_5 = forward_network()
    forward_model_6 = forward_network()

    # specify the forward model 1 and obtain the result of forward model 1
    forward_model_1.load_weights(args.forward_model_loc_1, by_name=False)
    forward_model_1.trainable = False
    predicted_y_before_transform_1 = forward_model_1(data_x_standarized, training=False)

    # specify the forward model 2 and obtain the result of forward model 2
    forward_model_2.load_weights(args.forward_model_loc_2, by_name=False)
    forward_model_2.trainable = False
    predicted_y_before_transform_2 = forward_model_2(data_x_standarized, training=False)

    # specify the forward model 3 and obtain the result of forward model 3
    forward_model_3.load_weights(args.forward_model_loc_3, by_name=False)
    forward_model_3.trainable = False
    predicted_y_before_transform_3 = forward_model_3(data_x_standarized, training=False)

    # specify the forward model 4 and obtain the result of forward model 4
    forward_model_4.load_weights(args.forward_model_loc_4, by_name=False)
    forward_model_4.trainable = False
    predicted_y_before_transform_4 = forward_model_4(data_x_standarized, training=False)

    # specify the forward model 5 and obtain the result of forward model 5
    forward_model_5.load_weights(args.forward_model_loc_5, by_name=False)
    forward_model_5.trainable = False
    predicted_y_before_transform_5 = forward_model_5(data_x_standarized, training=False)

    # specify the forward model 6 and obtain the result of forward model 6
    forward_model_6.load_weights(args.forward_model_loc_6, by_name=False)
    forward_model_6.trainable = False
    predicted_y_before_transform_6 = forward_model_6(data_x_standarized, training=False)

    # obtain the predicted data using inverse network
    predicted_y_points_1 = scaler2.inverse_transform(predicted_y_before_transform_1)
    predicted_y_points_2 = scaler2.inverse_transform(predicted_y_before_transform_2)
    predicted_y_points_3 = scaler2.inverse_transform(predicted_y_before_transform_3)
    predicted_y_points_4 = scaler2.inverse_transform(predicted_y_before_transform_4)
    predicted_y_points_5 = scaler2.inverse_transform(predicted_y_before_transform_5)
    predicted_y_points_6 = scaler2.inverse_transform(predicted_y_before_transform_6)

    # select the best prediction and evaluate using our defined metrics
    predicted_y_points = []
    for i in range(data_y.shape[0]):
        inverse_design_predicted_error_1 = K.sum(K.abs(predicted_y_points_1[i, :] - data_y[i, :]))
        inverse_design_predicted_error_2 = K.sum(K.abs(predicted_y_points_2[i, :] - data_y[i, :]))
        inverse_design_predicted_error_3 = K.sum(K.abs(predicted_y_points_3[i, :] - data_y[i, :]))
        inverse_design_predicted_error_4 = K.sum(K.abs(predicted_y_points_4[i, :] - data_y[i, :]))
        inverse_design_predicted_error_5 = K.sum(K.abs(predicted_y_points_5[i, :] - data_y[i, :]))
        inverse_design_predicted_error_6 = K.sum(K.abs(predicted_y_points_6[i, :] - data_y[i, :]))
        error_ratio_1 = tf.cast(inverse_design_predicted_error_1, dtype=tf.float32) / (K.sum(data_y[i, :]))
        error_ratio_2 = tf.cast(inverse_design_predicted_error_2, dtype=tf.float32) / (K.sum(data_y[i, :]))
        error_ratio_3 = tf.cast(inverse_design_predicted_error_3, dtype=tf.float32) / (K.sum(data_y[i, :]))
        error_ratio_4 = tf.cast(inverse_design_predicted_error_4, dtype=tf.float32) / (K.sum(data_y[i, :]))
        error_ratio_5 = tf.cast(inverse_design_predicted_error_5, dtype=tf.float32) / (K.sum(data_y[i, :]))
        error_ratio_6 = tf.cast(inverse_design_predicted_error_6, dtype=tf.float32) / (K.sum(data_y[i, :]))
        best_flag = np.argmin(
            [error_ratio_1, error_ratio_2, error_ratio_3, error_ratio_4, error_ratio_5, error_ratio_6])
        if best_flag == 0:
            predicted_y_points.append(predicted_y_points_1[i, :])
        elif best_flag == 1:
            predicted_y_points.append(predicted_y_points_2[i, :])
        elif best_flag == 2:
            predicted_y_points.append(predicted_y_points_3[i, :])
        elif best_flag == 3:
            predicted_y_points.append(predicted_y_points_4[i, :])
        elif best_flag == 4:
            predicted_y_points.append(predicted_y_points_5[i, :])
        else:
            predicted_y_points.append(predicted_y_points_6[i, :])


    predicted_y_points = np.array(predicted_y_points)
    predicted_points_df = pd.DataFrame(predicted_y_points)
    predicted_points_df.to_excel('result-out of boundary.xlsx', index=False)  # the name can be changed for different purposes


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.train_path = "train.xlsx"
    args.test_path = "test-out of boundary.xlsx"
    args.inverse_model_loc_1 = "inverse0.keras"
    args.inverse_model_loc_2 = "inverse11.keras"
    args.inverse_model_loc_3 = "inverse10.keras"
    args.inverse_model_loc_4 = "inverse6.keras"
    args.inverse_model_loc_5 = "inverse14.keras"
    args.inverse_model_loc_6 = "inverse16.keras"
    args.forward_model_loc_1 = "forward2.keras"
    args.forward_model_loc_2 = "forward12.keras"
    args.forward_model_loc_3 = "forward0.keras"
    args.forward_model_loc_4 = "forward3.keras"
    args.forward_model_loc_5 = "forward14.keras"
    args.forward_model_loc_6 = "forward6.keras"
    # args.forward_model_loc_1 = "forward0_4para.keras"
    # args.forward_model_loc_2 = "forward1_4para.keras"
    # args.forward_model_loc_3 = "forward2_4para.keras"
    # args.forward_model_loc_4 = "forward3_4para.keras"
    # args.forward_model_loc_5 = "forward4_4para.keras"
    # args.forward_model_loc_6 = "forward5_4para.keras"
    main(args)
