import * as tf from '@tensorflow/tfjs';
import parse from 'csv-parse/lib/sync';
import fs from 'fs';
var math = require('mathjs');

// import assert from 'assert';
// const tf = require('@tensorflow/tfjs');
// const parse = require('csv-parse');
// const fs = require('fs');

class Config {}
Config.epochs = 200;
Config.batch_size = 7;
Config.window_size = 7;
Config.hidden_layer_size = 1024;
Config.number_of_layers = 1;
Config.number_of_classes = 1;
Config.learning_rate = 1;
Config.optimizer = tf.train.rmsprop(Config.learning_rate);


class DataFeed {
  constructor(csvPath) {
    this.csvPath = csvPath;
  }

  loadData() {
    const data = fs.readFileSync('tesla_stocks.csv', 'utf8');
    return this.parseData(data);
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
    while (index < this.data.length - windowSize) {
      const row = this.data.slice(index, index + windowSize - 1);
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
  }

  standardscaler() {
    this.mean = math.mean(this.data);
    this.stdev = math.std(this.data);
    const results = [];
    for (let i = 0; i<this.data.length; i++) {
      const result = (this.data[i] - this.mean)/this.stdev;
      results.push(result);
    }
    return results;
  }
  unscale(data) {
    const results = [];
    for (let i=0; i<data.length; i++) {
      results.push(data[i] * this.stdev + this.mean);
    }
    return results;
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

    model.add(tf.layers.lstm({
      units: Config.hidden_layer_size,
      recurrentInitializer: 'glorotNormal',
      inputShape: [Config.window_size - 1, 1],
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
      activation: 'relu6'
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

    let samples = tf.reshape(tf.tensor2d(this.x_train), [this.x_train.length, Config.window_size - 1, 1]);
    let labels = tf.reshape(tf.tensor(this.y_train), [this.y_train.length, 1]);
    let test_data = tf.reshape(tf.tensor2d(this.x_test), [this.x_test.length, Config.window_size - 1, 1]);
    let test_labels = tf.reshape(tf.tensor1d(this.y_test), [this.y_test.length, 1]);

    // Config.epochs = 5;
    // for (let itr = 0; itr<Config.epochs; itr++) {
      const history = await model.fit(samples, labels, {
        epochs: 10,
        batchSize: 64,
        validationData: [test_data, test_labels],
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
    const dataFeed = new DataFeed('tesla_stocks.csv');
    const data = dataFeed.loadData().map(record =>  record.Close);
    const scaler = new Scaler(data);
    const closingPrices = scaler.standardscaler();
    console.log("Total numbner of days in the data set: " + closingPrices.length);

    // 1. data processing
    // a. find a way to scale the closing prices
    // b. striding
    const results = new DataFormatter(closingPrices).stride(7);
    // assert(results.training.length != results.label.length);
    const x_train = results.training.slice(0, 699);
    const x_test = results.training.slice(700, 749);
    const y_train = results.label.slice(0, 699);
    const y_test = results.label.slice(700, 749);

    const intelli = new Intelligence(x_train, y_train, x_test, y_test);
    (async function(intelli, x_test, scaler) {
      console.log("Intelli-training");
      const model = await intelli.train();

      // predicting
      const index = Math.floor(Math.random() * 1000 % x_test.length);
      const sample = tf.reshape(tf.tensor1d(x_test[index]), [1, Config.window_size - 1, 1]);
      const result = model.predict(sample);
      console.log('Sample: '+ scaler.unscale(x_test[index]));      
      console.log('Prediction: ' + scaler.unscale([result.dataSync()]));
      console.log('Expectation: ' + scaler.unscale([y_test[index]]));
      
    })(intelli, x_test, scaler);
  }
}

document.getElementById('mainButton').addEventListener('click', () => {
  MainDriver.main();
});

