from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('weights', None,
                    'path to weights file', short_name='w')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny', short_name='t')
flags.DEFINE_string('output', None, 'path to saved_model', short_name='o')
flags.DEFINE_string('classes', None, 'path to classes file', short_name='c')
flags.DEFINE_string('image', None, 'path to input image', short_name='i')
flags.DEFINE_integer('num_classes', None, 'number of classes in the model', short_name='n')

flags.mark_flags_as_required(['weights', 'output', 'classes', 'num_classes', 'image'])


def main(_argv):
    import time

    import tensorflow as tf

    from yolov3_tf2.dataset import transform_images
    from yolov3_tf2.models import (
        YoloV3, YoloV3Tiny
    )

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    tf.saved_model.save(yolo, FLAGS.output)
    logging.info("model saved to: {}".format(FLAGS.output))

    model = tf.saved_model.load(FLAGS.output)
    infer = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    logging.info(infer.structured_outputs)

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    img = tf.image.decode_image(open(FLAGS.image, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img = transform_images(img, 416)

    t1 = time.time()
    outputs = infer(img)
    boxes, scores, classes, nums = outputs["yolo_nms"], outputs["yolo_nms_1"], \
                                   outputs["yolo_nms_2"], outputs["yolo_nms_3"]
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           scores[0][i].numpy(),
                                           boxes[0][i].numpy()))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
