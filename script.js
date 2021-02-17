// https://github.com/ohyicong/masksdetection/blob/master/web_demo/templates/index.html
// https://github.com/tensorflow/tfjs-examples
// https://github.com/tensorflow/tfjs-models/tree/master/blazeface
// https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Manipulating_video_using_canvas

const state = {backend: 'webgl'}; // 'wasm' (install additional dependencies), 'webgl', 'cpu'
const videoWidth=720, videoHeight=560;
const videoConstraints = {
  audio: false,
  video: { 
    width: videoWidth, height: videoHeight,  
    frameRate: { ideal: 10, max: 21 },
    facingMode: 'user'  //facingMode determines which camera on a mobile (user=front,environment=back)
  }
}

let faceModel, maskModel;

const video = document.getElementById('video');
const spinner = document.getElementById('spinner')

// setup canvas:
const canvas = document.getElementById('output');
canvas.width = videoWidth;
canvas.height = videoHeight;
const ctx = canvas.getContext('2d');




async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia(videoConstraints);
  video.srcObject = stream;
    return new Promise((resolve) => {
      video.onloadedmetadata = () => {
        resolve(video);
      };
  });
}

async function renderPrediction() {
  
  tf.engine().startScope() 
  try {
    //? AnontateBoxes => detects eyes, ears, nose, & mouth & return locations.
    //? estimatefaces model takes in 4 parameter (1) video, returnTensors, flipHorizontal, and annotateBoxes
    
    const faces = await faceModel.estimateFaces(video, false, false, false);

    const offset = tf.scalar(127.5); // will be used to crop image

    console.log('Num of faces detected:', faces.length)
  
    ctx.clearRect(0, 0, canvas.width, canvas.height); //clear previous rectangle
    
    // if face(s) detected, detect mask and draw rect with class and probability: 
    if (faces.length) {     
        for (const face of faces){
          console.log('face:', face)

          // determine face box dimensions:
          const [topLeftX, topLeftY] = face.topLeft; 
          const [bottomRightX, bottomRightY] = face.bottomRight;
          const [boxWidth, boxHeight] = [bottomRightX - topLeftX, bottomRightY - topLeftY];
          
          // crop image to the face box only:
          let inputImage = tf.browser.fromPixels(video).toFloat()
          inputImage = inputImage.sub(offset).div(offset);
          inputImage = inputImage.slice([parseInt(topLeftY),parseInt(topLeftX),0],[parseInt(boxHeight),parseInt(boxWidth),3])
          inputImage = inputImage.resizeBilinear([224,224]).reshape([1,224,224,3])

          // get mask face confidence results:
          let [maskConfidence, noMaskConfidence] = await maskModel.predict(inputImage).dataSync()
          maskConfidence = (maskConfidence*100).toPrecision(3).toString() //change to percentage(%)
          noMaskConfidence = (noMaskConfidence*100).toPrecision(3).toString() //change to percentage(%)
          
          // draw rect on face:
          ctx.beginPath()
          if (noMaskConfidence > maskConfidence){ 
              ctx.strokeStyle="red"
              ctx.fillStyle = "red";
              text = "No Mask: "+noMaskConfidence+"%";
          }
          else{
              ctx.strokeStyle="green"
              ctx.fillStyle = "green";
              text = "Mask: "+maskConfidence+"%";
          }
          
          ctx.lineWidth = "3"
          ctx.rect(topLeftX, topLeftY, boxWidth, boxHeight) //draw box in canvas
          ctx.stroke() //draw box in canvas

          ctx.font = "bold 15pt sans-serif";
          ctx.fillText(text,topLeftX+5,topLeftY+20) //draw text in canvas

          video.classList.remove('hidden')
          spinner.classList.add('hidden')
        }

    }

  } 
  catch (err) {
    console.error(err)
  }

  tf.engine().endScope() 
  window.requestAnimationFrame(renderPrediction);
};


// self invoked function
(async function setupPage() {
  // load models and prepare TensorFlow in parallel:
  [faceModel, maskModel] = await Promise.all([blazeface.load(), tf.loadLayersModel('./model/model.json'), tf.setBackend(state.backend)]);
  console.log("face model loaded",faceModel);
  console.log("mask model loaded",maskModel);
  
  await setupCamera();
  
  renderPrediction();
})();

