const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const sharp = require('sharp')
console.log("Hola tensor!")

async function readImage(path) {
    const imageBuffer = fs.readFileSync(path);
    imageBufferResized = await sharp(imageBuffer).resize(100,100).toBuffer()
    const tfimage = tf.node.decodeImage(imageBufferResized);
    return tfimage;
}

async function detect() {
    let labels = ['Vicente', 'Sebastian', 'Paulo'];
    let model = await tf.loadLayersModel('file://model/model.json');
    model.summary();
    tfimage = await readImage('images/grande.png');
    console.log(tfimage);
    tfimage = tfimage.div(255.0)
   // tfimage.print();
    newtensor = tfimage.expandDims()
    result = model.predict(newtensor);
    result.print()
    result.argMax(1).dataSync(


    ).map( m => console.log(labels[m]));

}

detect();