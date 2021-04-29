import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import cv2
import time
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import numpy as np
import warnings

flags.DEFINE_string('video', './data/video/terrace1.mp4', 'path to input video')
flags.DEFINE_string('output', './data/output/output_video.avi', 'path to output video')

warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
tf.get_logger().setLevel('ERROR')   # Suppress TensorFlow logging (2)


@tf.function
def detect_fn(detection_model, image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections


def main(_argv):

    PATH_TO_LABELS = './ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/mscoco_label_map.pbtxt'

    PATH_TO_CFG = './ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config'
    PATH_TO_CKPT = './ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint'

    print('Loading model... ', end='')
    start_time = time.time()

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)

    # video_path = './data/video/terrace1.mp4'
    vid = cv2.VideoCapture(FLAGS.video)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    print('Running inference for {}... '.format(FLAGS.video), end='\n')
    frame_id = 0

    while(True):
        print('farame:', frame_id)
        ret, frame = vid.read()

        if not ret:
            print('Video has ended or failed, try a different video format!')
            break

        input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)
        detections = detect_fn(detection_model, input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            
        label_id_offset = 1
        image_np_with_detections = frame.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.30,
                agnostic_mode=False)

        cv2.imshow('frame', image_np_with_detections)
        out.write(image_np_with_detections)
        frame_id += 1

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
