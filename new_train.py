import tensorflow as tf

from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)

import yolov3_tf2.dataset as dataset


def main():
    batch_size = 4
    input_size = 416
    num_classes = 10
    shuffle_buffer_size = 128

    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks

    # load train set
    train_dataset = dataset.load_tfrecord_dataset("C:/Users/14841/Desktop/tf_out/train/yymnist.tfrecord",
                                                  "C:/Users/14841/Desktop/tf_out/yymnist.names",
                                                  input_size)
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer_size)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, input_size),
        dataset.transform_targets(y, anchors, anchor_masks, input_size)))

    # load valid set
    val_dataset = dataset.load_tfrecord_dataset("C:/Users/14841/Desktop/tf_out/valid/yymnist.tfrecord",
                                                "C:/Users/14841/Desktop/tf_out/yymnist.names",
                                                input_size)
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, input_size),
        dataset.transform_targets(y, anchors, anchor_masks, input_size)))

    model = YoloV3(input_size, training=True, classes=num_classes)

    learning_rate = 0.001

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss = [YoloLoss(anchors[mask], classes=num_classes)
            for mask in anchor_masks]

    model.compile(optimizer=optimizer, loss=loss)
    model.summary()
    epochs = 10

    train_size = 1000
    valid_size = 200

    history = model.fit(train_dataset,verbose=True,
                        steps_per_epoch=train_size // batch_size,
                        epochs=epochs,
                        validation_steps=valid_size // batch_size,
                        validation_data=val_dataset)


if __name__ == '__main__':
    main()