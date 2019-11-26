const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
console.log("Hola tensor!")

function readImage(path) {
    const imageBuffer = fs.readFileSync(path);
    const tfimage = tf.node.decodeImage(imageBuffer);
    return tfimage;
}

async function detect() {
    let labels = ['Vicente', 'Sebastian', 'Paulo'];
    let model = await tf.loadLayersModel('file://model/model.json');
    model.summary();
    tfimage = readImage('images/vicente2.jpg');
    tfimage = tfimage.div(255.0)
    tfimage.print();
    newtensor = tfimage.expandDims()
    result = model.predict(newtensor);
    result.print()
    result.argMax(1).dataSync(


    ).map( m => console.log(labels[m]));

}

detect();