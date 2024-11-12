#Supercombo model from: https://github.com/commaai/openpilot/tree/master
#Pure model code modified from: https://github.com/MTammvee/openpilot-supercombo-model/blob/main/openpilot_onnx.py


import cv2
import json
import numpy as np
import onnxruntime
import pandas as pd

import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt

X_IDXS = np.array([0., 0.1875, 0.75, 1.6875, 3., 4.6875,
                   6.75, 9.1875, 12., 15.1875, 18.75, 22.6875,
                   27., 31.6875, 36.75, 42.1875, 48., 54.1875,
                   60.75, 67.6875, 75., 82.6875, 90.75, 99.1875,
                   108., 117.1875, 126.75, 136.6875, 147., 157.6875,
                   168.75, 180.1875, 192.])

def parse_image(frame):
    H = (frame.shape[0] * 2) // 3
    W = frame.shape[1]
    parsed = np.zeros((6, H // 2, W // 2), dtype=np.uint8)

    parsed[0] = frame[0:H:2, 0::2]
    parsed[1] = frame[1:H:2, 0::2]
    parsed[2] = frame[0:H:2, 1::2]
    parsed[3] = frame[1:H:2, 1::2]
    parsed[4] = frame[H:H + H // 4].reshape((-1, H // 2, W // 2))
    parsed[5] = frame[H + H // 4:H + H // 2].reshape((-1, H // 2, W // 2))

    return parsed

def separate_points_and_std_values(df):
    points = df.iloc[lambda x: x.index % 2 == 0]
    std = df.iloc[lambda x: x.index % 2 != 0]
    points = pd.concat([points], ignore_index=True)
    std = pd.concat([std], ignore_index=True)

    return points, std

def main():
    model = "Supercombo/supercombo.onnx"
    session = onnxruntime.InferenceSession(model, None)


    image_path = "Supercombo/frame_6.jpg"
    frame = cv2.imread(image_path)

    if frame is None:
        print("Error: Image not found.")
        return

    width, height = 512, 256
    dim = (width, height)

    img = cv2.resize(frame, dim)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
    parsed = parse_image(img_yuv)

    parsed_arr = np.array([parsed, parsed])  # Duplicate for input size
    parsed_arr.resize((1, 12, 128, 256))

    data = json.dumps({'data': parsed_arr.tolist()})
    data = np.array(json.loads(data)['data']).astype('float32')

    input_imgs = session.get_inputs()[0].name
    desire = session.get_inputs()[1].name
    initial_state = session.get_inputs()[2].name
    traffic_convention = session.get_inputs()[3].name
    output_name = session.get_outputs()[0].name

    desire_data = np.zeros((1, 8), dtype='float32')
    traffic_convention_data = np.zeros((1, 512), dtype='float32')
    initial_state_data = np.zeros((1, 2), dtype='float32')

    result = session.run([output_name], {
        input_imgs: data,
        desire: desire_data,
        traffic_convention: traffic_convention_data,
        initial_state: initial_state_data
    })

    res = np.array(result)

    lanes_start_idx = 4955
    lanes_end_idx = lanes_start_idx + 528
    road_start_idx = lanes_end_idx + 8
    road_end_idx = road_start_idx + 264

    lanes = res[:, :, lanes_start_idx:lanes_end_idx]
    lane_road = res[:, :, road_start_idx:road_end_idx]

    df_lanes = pd.DataFrame(lanes.flatten())

    points_ll_t, std_ll_t = separate_points_and_std_values(df_lanes[0:66])
    points_ll_t2, std_ll_t2 = separate_points_and_std_values(df_lanes[66:132])

    points_l_t, std_l_t = separate_points_and_std_values(df_lanes[132:198])
    points_l_t2, std_l_t2 = separate_points_and_std_values(df_lanes[198:264])

    middle = points_ll_t2.add(points_l_t, fill_value=0) / 2

    plt.scatter(middle, X_IDXS, color="g")
    plt.scatter(points_ll_t2, X_IDXS, color="y")
    plt.scatter(points_l_t, X_IDXS, color="y")
    
    plt.title("Road lines")
    plt.xlabel("Red - road lines | Green - predicted path | Yellow - lane lines")
    plt.ylabel("Range")
    #plt.show()
    plt.savefig('Supercombo/predicted_movement.png') 

if __name__ == "__main__":
    main()
