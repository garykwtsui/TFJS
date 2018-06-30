import * as tf from '@tensorflow/tfjs';
import parse from 'csv-parse/lib/sync';
import fs from 'fs';
import { sigmoid } from '@tensorflow/tfjs';
var math = require('mathjs');

class Config {}
Config.epochs = 10;
Config.batch_size = 7;
Config.window_size = 10;
Config.hidden_layer_size = 64;
Config.number_of_layers = 1;
Config.number_of_classes = 1;
Config.learning_rate = 1;
Config.validation_spit = 0.05;
Config.optimizer = tf.train.rmsprop(Config.learning_rate);

class Intelligence  {
    static getModel() {
        const model = tf.sequential({
            layers: [tf.layers.dense({
                units: Config.hidden_layer_size,
                // find out why  inputShape: [Config.window_size, 1] will give bad results
                inputShape: [Config.window_size], 
                activation: 'relu'
            })]
        });
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }));

        const optimizer = tf.train.adam(0.2);
        model.compile({
            optimizer: optimizer,
            loss: 'meanSquaredError',
            metrics: ['accuracy']
        });
        return model;
    }
}

class MainDriver {
    static main() {
        console.log("Hi");
        const data3 = {
            training: [
                [0, 0, 0], 
                [0, 0, 1],
                [0, 1, 0], 
                [0, 1, 1],
                [1, 0, 0], 
                [1, 0, 1],
                [1, 1, 0], 
                [1, 1, 1]
            ],
            label: [
                0, 
                1, 
                1, 
                0, 
                1,
                0,
                0,
                1
            ]
        }

        const data2 = {
            training: [
                [0, 0], 
                [0, 1],
                [1, 0], 
                [1, 1],

            ],
            label: [
                0, 
                1, 
                1, 
                0
            ]
        }
        const data = data3;

        Config.window_size = data.training[0].length;   
        const model = Intelligence.getModel();
        // const samples = tf.reshape(tf.tensor2d(data.training), [data.training.length, Config.window_size, 1]);
        // const labels = tf.reshape(tf.tensor(data.label), [data.label.length, 1]);
        const samples = tf.tensor2d(data.training);
        const labels = tf.tensor(data.label);
        (async function() {
                console.log("Going to train");
                const h = await model.fit(samples, labels, {
                    // batchSize: 1,
                    epochs: 100,
                    // validationSplit: 0.2,
                    callbacks: {
                        onEpochEnd: async (epoch, log) => {
                            // console.log(`Epoch ${epoch}: loss = JSON.stringify(${log})`);
                            console.log('Epoch (' + epoch + ') : ' + JSON.stringify(log));
                        }
                    }
                });
                console.log(JSON.stringify(h));

                // for (let k = 0; k < 10; k++) {
                //     const index = Math.floor(Math.random() * 1000 % data.training.length);
                //     const s = data.training[index];
                //     const sample = tf.tensor(s, [1, Config.window_size]);
                //     // const sample = tf.tensor(s);
                //     const result = model.predict(sample);
                //     console.log(s + " : " + result.dataSync());
                // }
                for (let k =0; k<data.training.length; k++) {
                    const s = data.training[k];
                    const result = model.predict(tf.tensor(s, [1, Config.window_size]));
                    // const result = model.predict(tf.tensor3d(s, [1, Config.window_size, 1]));
                    console.log(s + " : " + result.dataSync());
                }

        })();
    }
}

document.getElementById('mainButton').addEventListener('click', () => {
    MainDriver.main();
});