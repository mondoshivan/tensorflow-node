import * as tf from '@tensorflow/tfjs-node'

const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

model.compile({
  loss: 'meanSquaredError',
  optimizer: 'sgd',
  metrics: ['accuracy']
});

/**
 * Call model.summary() to print a useful summary of the model:
 * - Name and type of all layers in the model.
 * - Output shape for each layer.
 * - Number of weight parameters of each layer.
 * - If the model has general topology (discussed below), the inputs each layer receives
 * - The total number of trainable and non-trainable parameters of the model.
 */
model.summary();

/**
 * Trains a tf.sequential model using x and y values that satisfy the equation: y = 2x - 1
 */
const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

/**
 * Train the model using the data.
 * verbose = 1 would print real-time updates of loss and metric values and training speed
 */
const verbosity = 1
await model.fit(xs, ys, { epochs: 500, verbose: verbosity });

/**
 * Serialization
 * One of the major benefits of using a LayersModel over the lower-level API is the ability to save and load a model.
 * A LayersModel knows about:
 * - the architecture of the model, allowing you to re-create the model.
 * - the weights of the model
 * - the training configuration (loss, optimizer, metrics).
 * - the state of the optimizer, allowing you to resume training.
 */
// const saveResult = await model.save('localstorage://my-model-1');
// const model = await tf.loadLayersModel('localstorage://my-model-1');

const prediction = model.predict(tf.tensor2d([20], [1, 1])) as tf.Tensor
console.log(`preduction: ${prediction.dataSync()}`);