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
        const data = fs.readFileSync('sinwave.csv', 'utf8');
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

class Intelligence {
    constructor(x_train, y_train, x_test, y_test) {
        this.x_train = x_train;
        this.y_train = y_train;
        this.x_test = x_test;
        this.y_test = y_test;
    }

    buildModel() {
        // return this.buildSimpleModel();
        return this.buildComplexModel();
    }
    buildSimpleModel() {
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
    buildComplexModel() {

        const model = tf.sequential();

        model.add(tf.layers.lstm({
            units: Config.hidden_layer_size,
            recurrentInitializer: 'glorotNormal',
            inputShape: [Config.window_size, 1],
            returnSequences: false
        }));

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

    buildComplexModel_old() {
        const model = tf.sequential();

        model.add(tf.layers.lstm({
            units: Config.window_size,
            recurrentInitializer: 'glorotNormal',
            inputShape: [Config.window_size, 1],
            returnSequences: true
        }));

        model.add(tf.layers.dropout(0.2));

        model.add(tf.layers.lstm({
            units: Config.hidden_layer_size,
            recurrentInitializer: 'glorotNormal',
            returnSequences: false
        }));

        model.add(tf.layers.dropout(0.2));

        model.add(tf.layers.dense({
            units: 1
        }));

        model.add(tf.layers.activation({
            activation: 'linear'
        }));
        const optimizer = tf.train.adam(0.2);

        model.compile({
            loss: 'meanSquaredError',
            optimizer: optimizer,
            metrics: ['accuracy']
        });

        return model;
    }


    async train() {
        // setup model
        const model = this.buildModel();

        let samples = tf.reshape(tf.tensor2d(this.x_train), [this.x_train.length, Config.window_size, 1]);
        let labels = tf.reshape(tf.tensor(this.y_train), [this.y_train.length, 1]);
        let test_data = tf.reshape(tf.tensor2d(this.x_test), [this.x_test.length, Config.window_size, 1]);
        let test_labels = tf.reshape(tf.tensor1d(this.y_test), [this.y_test.length, 1]);

        console.log("Going to train now");
        // for (let itr = 0; itr<Config.epochs; itr++) {
        const history = await model.fit(samples, labels, {
            epochs: Config.epochs,
            shuffle: false,
            // batchSize: 512,
            validationSplit: 0.05,
            // validationData: [test_data, test_labels],
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
        console.log(JSON.stringify(history));

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
        MainDriver.stockMain();
    }

    static testMain() {
        console.log("Hello Gary!");
        (async function () {
            const model = tf.sequential({
                layers: [tf.layers.dense({
                    units: 1,
                    inputShape: [10, 1]
                })]
            });
            model.add(tf.layers.flatten());
            model.add(tf.layers.dense({
                units: 1,
            }));

            model.compile({
                optimizer: 'sgd',
                loss: 'meanSquaredError',
                metrics: ['accuracy']
            });

            // const samples = tf.ones([100, 10]);
            // const labels = tf.ones([100, 1]);
            const samples = tf.randomNormal([100, 10, 1]);
            const labels = tf.ones([100, 1]);


            for (let i = 1; i < 5; ++i) {
                console.log("iteration: " + i);
                const h = await model.fit(samples, labels, {
                    batchSize: 4,
                    epochs: 3,
                    validationSplit: 0.1
                });
                console.log(JSON.stringify(h));
                console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
            }
        })();
    }

    static stockMain() {
        // 0. loading data
        const dataFeed = new DataFeed('sinwave.csv');
        const data = dataFeed.loadData('\n');
        const scaler = new Scaler(data);
        const closingPrices = scaler.passthruScaler();
        console.log("Total numbner of days in the data set: " + data.length);

        const results = new DataFormatter(data).stride(Config.window_size);
        const length = results.training.length;
        // const trainingSize = Math.floor(length * 0.8);
        // const testingSize = Math.floor(length - trainingSize);

        const trainingSize = length;
        const x_train = results.training.slice(0, trainingSize);
        const x_test = results.training.slice(length - (length * 0.1), length);
        const y_train = results.label.slice(0, trainingSize);
        const y_test = results.label.slice(length - (length * 0.1), length);

        const intelli = new Intelligence(x_train, y_train, x_test, y_test);
        (async function (intelli, x_test, scaler) {
            console.log("Intelli-training");
            const model = await intelli.train();
            //predicting all on a point by point basis.
            var outputStr = '';
            for (let k = 0; k < results.training.length; k++) {
                const test = results.training.slice(0, trainingSize);
                const sample = tf.reshape(tf.tensor1d(test[k]), [1, Config.window_size, 1]);
                const label = results.label[k];
                const result = model.predict(sample);
                // console.log(result.dataSync() +', '+ label);
                outputStr += result.dataSync() + ', ' + label + '\n';
            }
            console.log(outputStr);

        })(intelli, x_test, scaler);
    }
}

document.getElementById('mainButton').addEventListener('click', () => {
    MainDriver.main();
});