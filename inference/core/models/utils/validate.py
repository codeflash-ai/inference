def get_num_classes_from_model_prediction_shape(len_prediction, masks=0, keypoints=0):
    # Subtract constants and variables in a single operation for efficiency
    return len_prediction - 5 - masks - keypoints * 3
