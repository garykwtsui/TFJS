import * as tf from '@tensorflow/tfjs';
import parse from 'csv-parse/lib/sync';
import fs from 'fs';
var math = require('mathjs');

class Config {}
Config.epochs = 70;
Config.window_size = 50;
Config.hidden_layer_size = 256;

class DataFeed {
    constructor(csvPath) {
        this.csvPath = csvPath;
    }

    loadData(delimiter) {
        console.log("Going to read: " + this.csvPath);
        // TODO:    looks like fs.readFileSync reads a string at 'build-time' instead of run-time, so passing a param in
        //          will not work
        const data = fs.readFileSync('sinewave.csv', 'utf8');
        if (!delimiter) {
            return this.parseData(data);
        } else {
            return data.split(delimiter);
        }
    }

    parseData(data) {
        if (!data) {
            throw "no data";
        }
        var records = parse(data, {
            columns: true
        });
        return records;
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

class Intelligence {
    constructor(x_train, y_train, x_test, y_test) {
        this.x_train = x_train;
        this.y_train = y_train;
        this.x_test = x_test;
        this.y_test = y_test;
    }

    buildModel() {
        const model = tf.sequential();

        model.add(tf.layers.dense({
            units: Config.hidden_layer_size,
            inputShape: [Config.window_size, 1],
        }));

        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({
            units: 1,
            activation: 'linear'
        }));

        model.compile({
            loss: 'meanSquaredError',
            optimizer: 'rmsprop',
            metrics: ['accuracy']
        });

        return model;
    }

    async train() {
        // setup model
        const model = this.buildModel();

        let samples = tf.reshape(tf.tensor2d(this.x_train), [this.x_train.length, Config.window_size, 1]);
        let labels = tf.reshape(tf.tensor(this.y_train), [this.y_train.length, 1]);
        // let test_data = tf.reshape(tf.tensor2d(this.x_test), [this.x_test.length, Config.window_size, 1]);
        // let test_labels = tf.reshape(tf.tensor1d(this.y_test), [this.y_test.length, 1]);

        console.log("Going to train now");
        const history = await model.fit(samples, labels, {
            epochs: Config.epochs,
            shuffle: false,
            validationSplit: 0.05,
            callbacks: {
                onEpochBegin: async (epoch, log) => {
                    console.log(`Epoch ${epoch} started.`);
                },
                onEpochEnd: async (epoch, log) => {
                    console.log(`Epoch ${epoch}: loss = ${log.loss}`);
                }
            }
        });
        this.outputModelAccuracy(history);
        return model;
    }

    outputModelAccuracy(history) {
        const trainLoss = history.history['loss'][0];
        const trainAccuracy = history.history['acc'][0];
        const valLoss = history.history['val_loss'][0];
        const valAccuracy = history.history['val_acc'][0];
        console.log('Train Loss: ' + trainLoss);
        console.log('Train Acc: ' + trainAccuracy);
        console.log('Val Loss: ' + valLoss);
        console.log('Val Acc: ' + valAccuracy);
    }
}

class MainDriver {
    static main() {
        // 0. loading data
        console.log("Loading data");
        const dataFeed = new DataFeed('sinwave.csv');
        const data = dataFeed.loadData('\n');

        // 1. striding
        console.log("Preprocessing data");
        const results = new DataFormatter(data).stride(Config.window_size);
        const length = results.training.length;

        // 2. prepping data sets
        console.log("Preparing data sets");
        const trainingSize = length;
        const x_train = results.training.slice(0, trainingSize);
        // const x_test = results.training.slice(length - (length * 0.1), length);
        const y_train = results.label.slice(0, trainingSize);
        // const y_test = results.label.slice(length - (length * 0.1), length);

        // 3. training and predicting
        const intelli = new Intelligence(x_train, y_train);
        (async function (intelli) {
            console.log("Going to train now -- may take a while");
            const model = await intelli.train();

            console.log("Going to predict now");
            //predicting all on a point by point basis.
            var outputStr = '';
            for (let k = 0; k < results.training.length; k++) {
                const test = results.training.slice(0, trainingSize);
                const sample = tf.reshape(tf.tensor1d(test[k]), [1, Config.window_size, 1]);
                const label = results.label[k];
                const result = model.predict(sample);
                outputStr += result.dataSync() + ', ' + label + '\n';
            }

            console.log("Outputting results -- may take a while");
            console.log(outputStr);
        })(intelli);
    }
}

document.getElementById('mainButton-sinewave').addEventListener('click', () => {
    MainDriver.main();
});

/* unused classes for references only */
class Scaler {
    constructor(data) {
        this.data = data;
        this.mean = 0;
        this.stdev = 0;
        this.unscaler = null;
    }

    standardScaler() {
        this.mean = math.mean(this.data);
        this.stdev = math.std(this.data);
        const results = [];
        for (let i = 0; i < this.data.length; i++) {
            const result = (this.data[i] - this.mean) / this.stdev;
            results.push(result);
        }

        this.unscaler = (function (data) {
            const results = [];
            for (let i = 0; i < data.length; i++) {
                results.push(data[i] * this.stdev + this.mean);
            }
            return results;
        });

        return results;
    }

    passthruScaler() {
        this.unscaler = (function (data) {
            return data;
        });
        return this.data;
    }

    unscale(data) {
        return this.unscaler(data);
    }
}