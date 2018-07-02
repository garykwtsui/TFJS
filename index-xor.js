import * as tf from '@tensorflow/tfjs';
import {
    sigmoid
} from '@tensorflow/tfjs';

class Config {}
Config.epochs = 100;
Config.window_size = 0; // to be updated later.
Config.hidden_layer_size = 64;

class Intelligence {
    static getModel() {
        const model = tf.sequential({
            layers: [tf.layers.dense({
                units: Config.hidden_layer_size,
                inputShape: [Config.window_size],
                activation: 'relu'
            })]
        });
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

        // const data2 = {
        //     training: [
        //         [0, 0],
        //         [0, 1],
        //         [1, 0],
        //         [1, 1],

        //     ],
        //     label: [
        //         0,
        //         1,
        //         1,
        //         0
        //     ]
        // }

        const data = data3; // using 3 col XOR.
        Config.window_size = data.training[0].length;
        const model = Intelligence.getModel();
        const samples = tf.tensor2d(data.training);
        const labels = tf.tensor(data.label);
        (async function () {
            console.log("Going to train");
            const h = await model.fit(samples, labels, {
                epochs: Config.epochs,
                callbacks: {
                    onEpochEnd: async (epoch, log) => {
                        console.log('Epoch (' + epoch + ') : ' + JSON.stringify(log));
                    }
                }
            });
            console.log(JSON.stringify(h));
            for (let k = 0; k < data.training.length; k++) {
                const s = data.training[k];
                const result = model.predict(tf.tensor2d(s, [1, Config.window_size]));
                console.log(s + " : " + result.dataSync());
            }
        })();
    }
}

document.getElementById('mainButton-xor').addEventListener('click', () => {
    MainDriver.main();
});