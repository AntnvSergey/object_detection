import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf
import cv2

import time
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import numpy as np
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

tf.get_logger().setLevel('ERROR')   # Suppress TensorFlow logging (2)


@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections


image_path = './data/images/kite.jpeg'
image = cv2.imread(image_path)


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


print('Running inference for {}... '.format(image_path), end='')

input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)

detections = detect_fn(input_tensor)

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
image_np_with_detections = image.copy()

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

cv2.imshow('kite', image_np_with_detections)
cv2.imwrite('./data/output/kite_detection.png', image_np_with_detections)
cv2.waitKey(0)
