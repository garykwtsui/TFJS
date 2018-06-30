import * as tf from '@tensorflow/tfjs';
import parse from 'csv-parse/lib/sync';
import fs from 'fs';
var math = require('mathjs');

class Config {}
Config.epochs = 10;
Config.batch_size = 7;
Config.window_size = 10;
Config.hidden_layer_size = 512;
Config.number_of_layers = 1;
Config.number_of_classes = 1;
Config.learning_rate = 1;
Config.validation_spit = 0.05;
Config.optimizer = tf.train.rmsprop(Config.learning_rate);

class DataGenerator {
    static getData(n, windowSize) {
        const data = [];
        for(let i =0; i<n; i++) {
            data.push(i);
        }

        return new DataFormatter(data).stride(windowSize);
    }
}

class DataFormatter {
    constructor(data) {
        this.data = data;
    }

    stride(windowSize) {
        var X = [];
        var Y = [];
        var index = 0;
        while (index < this.data.length - windowSize - 1) {
            const row = this.data.slice(index, index + windowSize);
            const label = this.data[index + windowSize];
            X.push(row);
            Y.push(label);
            index++;
        }
        return {
            training: X,
            label: Y
        };
    }
}

class Intelligence  {
    static getModel() {
        const model = tf.sequential({
            layers: [tf.layers.dense({
                units: Config.hidden_layer_size,
                inputShape: [Config.window_size],
                activation: 'sigmoid'
            })]
        });
        // model.add(tf.layers.flatten());
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
        const data = DataGenerator.getData(1000,Config.window_size);
        const model = Intelligence.getModel();
        // const samples = tf.reshape(tf.tensor2d(data.training), [data.training.length, Config.window_size, 1]);
        // const labels = tf.reshape(tf.tensor(data.label), [data.label.length, 1]);
        const samples = tf.tensor2d(data.training);
        const labels = tf.tensor(data.label);
        (async function() {
                console.log("Going to train");
                const h = await model.fit(samples, labels, {
                    // batchSize: 4,
                    epochs: 50,
                    validationSplit: 0.2,
                    callbacks: {
                        onEpochEnd: async (epoch, log) => {
                            // console.log(`Epoch ${epoch}: loss = JSON.stringify(${log})`);
                            console.log('Epoch (' + epoch + ') : ' + JSON.stringify(log));
                        }
                    }
                });
                console.log(JSON.stringify(h));

                for (let k = 0; k < 10; k++) {
                    const index = Math.floor(Math.random() * 1000 % data.training.length);
                    const s = data.training[index];
                    const sample = tf.tensor2d(s, [1, Config.window_size]);
                    // const sample = tf.tensor(s);
                    const result = model.predict(sample);
                    console.log(s + " : " + result.dataSync());
                }

        })();
    }
}

document.getElementById('mainButton').addEventListener('click', () => {
    MainDriver.main();
});