// Based on an original demo at https://github.com/tensorflow/magenta-demos/tree/master/sketch-rnn-js.
// See LICENSE for full attribution and license details

dl = deeplearn;
math = dl.ENV.math;

const sketch = function (p) {
  "use strict";

  const classList = [
    'ant',
    'ambulance',
    'angel',
    'alarm_clock',
    'antyoga',
    'backpack',
    'barn',
    'basket',
    'bear',
    'bee',
    'beeflower',
    'bicycle',
    'bird',
    'book',
    'brain',
    'bridge',
    'bulldozer',
    'bus',
    'butterfly',
    'cactus',
    'calendar',
    'castle',
    'cat',
    'catbus',
    'catpig',
    'chair',
    'couch',
    'crab',
    'crabchair',
    'crabrabbitfacepig',
    'cruise_ship',
    'diving_board',
    'dog',
    'dogbunny',
    'dolphin',
    'duck',
    'elephant',
    'elephantpig',
    'everything',
    'eye',
    'face',
    'fan',
    'fire_hydrant',
    'firetruck',
    'flamingo',
    'flower',
    'floweryoga',
    'frog',
    'frogsofa',
    'garden',
    'hand',
    'hedgeberry',
    'hedgehog',
    'helicopter',
    'kangaroo',
    'key',
    'lantern',
    'lighthouse',
    'lion',
    'lionsheep',
    'lobster',
    'map',
    'mermaid',
    'monapassport',
    'monkey',
    'mosquito',
    'octopus',
    'owl',
    'paintbrush',
    'palm_tree',
    'parrot',
    'passport',
    'peas',
    'penguin',
    'pig',
    'pigsheep',
    'pineapple',
    'pool',
    'postcard',
    'power_outlet',
    'rabbit',
    'rabbitturtle',
    'radio',
    'radioface',
    'rain',
    'rhinoceros',
    'rifle',
    'roller_coaster',
    'sandwich',
    'scorpion',
    'sea_turtle',
    'sheep',
    'skull',
    'snail',
    'snowflake',
    'speedboat',
    'spider',
    'squirrel',
    'steak',
    'stove',
    'strawberry',
    'swan',
    'swing_set',
    'the_mona_lisa',
    'tiger',
    'toothbrush',
    'toothpaste',
    'tractor',
    'trombone',
    'truck',
    'whale',
    'windmill',
    'yoga',
    'yogabicycle'
  ];

  let isWaitingForHallucination = false;

  // sketch_rnn model
  let model;
  let temperature = 0.25;
  const minimumSequenceLength = 5; // We don't bother with predictions when we have fewer than this many samples.
  const defaultModel = "cat";

  let lastCommittedModelState; // RNN state as of the last time the pen lifted
  let currentModelState; // the more ephemeral RNN state, built on lastCommittedModelState, adding in the current stroke as it evolves

  let lastMouseState = null;
  let startingMouseState = null;
  const epsilon = 2.0; // we ignore mouse movement under this threshold

  let simplifiedRawLines;
  let pendingRawLine;
  let pendingStrokeIndex;
  let strokes;

  p.colorMode(p.HSB, 100);
  const lineColor = p.color(0, 0, 0);
  const hallucinationLineColor = p.color(30, 40, 90, 100);
  const hallucinatedSampleCount = 25; // How many steps forward do we project?
  let drawingGraphics = null;
  let hallucinationGraphics = null;

  let timeSamples = [];

  // UI
  let screenWidth,
    screenHeight;
  const lineWidth = 2.0;
  const screenScaleFactor = 3.0;

  const init = function () {
    ModelImporter.set_init_model(model_raw_data);
    model = new SketchRNN(ModelImporter.get_model_data());

    screenWidth = p.windowWidth;
    screenHeight = p.windowHeight;

    // Wire up the UI controls.
    document
      .getElementById("clearButton")
      .onclick = onClear;
    const selectElement = document.getElementById("model");
    for (let i = 0; i < classList.length; i++) {
      const optionElement = document.createElement("option");
      const formattedLabel = classList[i].replace("_", " ");
      optionElement.innerHTML = formattedLabel
        .charAt(0)
        .toUpperCase() + formattedLabel.slice(1);
      optionElement.setAttribute("value", classList[i]);
      if (classList[i] === defaultModel) {
        optionElement.setAttribute("selected", true);
      }
      selectElement.appendChild(optionElement);
    }
    selectElement.onchange = onModelSelection;
  };

  const disposeModelState = function (modelState) {
    if (modelState) {
      for (let component of modelState) {
        component.dispose();
      }
    }
  }

  const currentScaleFactor = function () {
    return model
      .get_info()
      .scale_factor / screenScaleFactor;
  }

  const scaleScreenSpaceStrokeSample = function (screenSpaceStrokeSample) {
    const scaleFactor = currentScaleFactor();
    return [
      screenSpaceStrokeSample[0] / scaleFactor,
      screenSpaceStrokeSample[1] / scaleFactor,
      screenSpaceStrokeSample[2]
    ];
  }

  const setIsLoading = function (isLoadingState) {
    const loadingElement = document.getElementById("loading");
    if (/Safari/.test(window.navigator.userAgent) && !/Chrome/.test(window.navigator.userAgent)) {
      loadingElement.innerHTML = "<p>Alas, Safari is unsupported; try Chrome?</p>"
    } else {
      document.getElementById("loading").style.opacity = isLoadingState ? 1 : 0;
    }
  }

  let lastHallucinationStartPoint = null;

  const restart = function () {
    // make sure we enforce some minimum size of our demo
    screenWidth = Math.max(window.innerWidth, 480);
    screenHeight = Math.max(window.innerHeight, 320);

    // variables for the sketch input interface.
    simplifiedRawLines = [];
    pendingRawLine = [];
    strokes = [];
    pendingStrokeIndex = null;
    lastMouseState = null;
    startingMouseState = null;

    lastHallucinationStartPoint = null;

    disposeModelState(lastCommittedModelState);
    lastCommittedModelState = null;
    disposeModelState(currentModelState);
    currentModelState = null;
  };

  const clearScreen = function () {
    p.background(0, 0, 100, 100);
    hallucinationGraphics.background(0, 0, 100, 100);
    drawingGraphics.clear();
  };

  p.setup = function () {
    init();

    const mainCanvasPixelDensity = p.pixelDensity();
    hallucinationGraphics = p.createGraphics(screenWidth * mainCanvasPixelDensity, screenHeight * mainCanvasPixelDensity);
    drawingGraphics = p.createGraphics(screenWidth * mainCanvasPixelDensity, screenHeight * mainCanvasPixelDensity);
    const prepareRenderer = (renderer) => {
      renderer.pixelDensity(1);
      renderer.scale(mainCanvasPixelDensity, mainCanvasPixelDensity);
      renderer.colorMode(p.HSB, 100);
    }
    prepareRenderer(hallucinationGraphics)
    prepareRenderer(drawingGraphics)

    restart();
    p.createCanvas(screenWidth, screenHeight);
    p.frameRate(60);
    clearScreen();

    // Preheat the shader stages used in the hallucination pipeline: it takes a
    // second or two the first time.
    hallucinate(model.update(model.zero_input(), model.zero_state()), [model.zero_input()]);

    console.log('ready.');
    setIsLoading(false);
  };

  const updateModelStateUsingCurrentStrokes = (isFinished) => {
    // We smooth the user input to reduce the number of input samples to feed
    // forward through the model.
    const simplifiedPendingRawLine = DataTool.simplify_line(pendingRawLine);
    if (simplifiedPendingRawLine.length <= 1) {
      return;
    }

    // Have we recorded any simplified lines for this stroke yet? Update if so, append if not.
    if (pendingStrokeIndex !== null) {
      simplifiedRawLines[simplifiedRawLines.length - 1] = simplifiedPendingRawLine;
    } else {
      simplifiedRawLines.push(simplifiedPendingRawLine);
    }

    // Where did the previous stroke end? For the first stroke: where did it all
    // begin? We need this because the network is trained on all relative motion.
    let previousStrokeFinalX = startingMouseState.x,
      previousStrokeFinalY = startingMouseState.y;
    if (strokes.length > 0) {
      const lastCommittedRawLineIndex = simplifiedRawLines.length - (lastMouseState.down
        ? 2
        : 1);
      if (lastCommittedRawLineIndex >= 0) {
        const lastCommittedPoint = simplifiedRawLines[lastCommittedRawLineIndex][simplifiedRawLines[lastCommittedRawLineIndex].length - 1];
        previousStrokeFinalX = lastCommittedPoint[0];
        previousStrokeFinalY = lastCommittedPoint[1];
      }
    }

    // Convert that smoothed stroke to the format the model expects and update our
    // internal state.
    const stroke = DataTool.line_to_stroke(simplifiedPendingRawLine, [
      previousStrokeFinalX, previousStrokeFinalY
    ], isFinished);
    if (pendingStrokeIndex !== null) {
      strokes = strokes
        .slice(0, pendingStrokeIndex)
        .concat(stroke);
    } else {
      pendingStrokeIndex = strokes.length;
      strokes = strokes.concat(stroke);
    }

    // Update our RNN state with the new strokes.
    if (strokes.length > minimumSequenceLength) {
      disposeModelState(currentModelState);
      if (lastCommittedModelState) {
        currentModelState = model.copy_state(lastCommittedModelState);
      } else {
        currentModelState = model.zero_state();
      }

      if (pendingStrokeIndex === 0) {
        currentModelState = model.update(model.zero_input(), currentModelState);
      }
      // Encode each sample in the latest stroke.
      for (let i = pendingStrokeIndex; i < strokes.length - 1; i++) {
        currentModelState = model.update(scaleScreenSpaceStrokeSample(strokes[i]), currentModelState);
      }

      // If the pen was just lifted, copy the pending model state onto the base state.
      if (strokes[strokes.length - 1][2] === 1) {
        disposeModelState(lastCommittedModelState);
        lastCommittedModelState = model.copy_state(currentModelState);
      }
    }
  }

  const hallucinate = function (modelState, strokes) {
    if (isWaitingForHallucination) {
      return;
    }
    isWaitingForHallucination = true;

    const lastStroke = strokes[strokes.length - 1];
    let lastSample = scaleScreenSpaceStrokeSample(lastStroke);
    let concatenatedSamples = null;
    let sampleCount = 0;

    let hallucinatedState = model.copy_state(modelState);
    while (sampleCount < hallucinatedSampleCount) {
      math.scope((keep, track) => {
        const oldModelState = hallucinatedState;
        hallucinatedState = model.update(lastSample, hallucinatedState);
        disposeModelState(oldModelState);
        hallucinatedState.forEach(keep);

        const modelPDF = model.get_pdf(hallucinatedState);
        const output = model.sample(modelPDF, temperature);

        if (concatenatedSamples) {
          const oldConcatenatedSample = concatenatedSamples;
          concatenatedSamples = keep(math.concat2D(concatenatedSamples, output.as2D(1, output.size), 0));
          oldConcatenatedSample.dispose();
        } else {
          concatenatedSamples = keep(output.as2D(1, output.size));
        }
        if (lastSample instanceof dl.Array1D) {
          lastSample.dispose();
        }
        lastSample = keep(output);
        sampleCount += 1;
      })
    }
    disposeModelState(hallucinatedState);
    lastSample.dispose();

    const startTime = Date.now();
    concatenatedSamples
      .data()
      .then((data) => {
        concatenatedSamples.dispose();

        // Process performance logs.
        const dt = Date.now() - startTime;
        timeSamples.push(dt);
        if (timeSamples.length > 100) {
          timeSamples.sort((a, b) => a - b);
          console.log(`Median: ${timeSamples[Math.ceil(timeSamples.length / 2)]}; Min: ${timeSamples[0]}; Max: ${timeSamples[timeSamples.length - 1]}`);
          timeSamples = [];
        }
        isWaitingForHallucination = false;

        if (!currentModelState) {
          // If we've reset since the GPU request was made, bail. This is only a weak
          // heuristic, rather than a rigorous queue/sequencing, but it's fine for this
          // demo.
          return;
        }

        // Find the point where the hallucination should start.
        const lastRawLineIndex = simplifiedRawLines.length - 1;
        const lastPoint = simplifiedRawLines[lastRawLineIndex][simplifiedRawLines[lastRawLineIndex].length - 1];
        let hallucinationX = lastPoint[0];
        let hallucinationY = lastPoint[1];

        // Fade out the previous hallucinations according to how far the user's moved.
        if (lastHallucinationStartPoint) {
          const dx = lastMouseState.x - lastHallucinationStartPoint[0];
          const dy = lastMouseState.y - lastHallucinationStartPoint[1];
          const drawingLengthSinceLastHallucination = lastMouseState.down
            ? Math.sqrt(dx * dx + dy * dy)
            : 10;
            hallucinationGraphics.background(0, 0, 100, p.lerp(10, 90, drawingLengthSinceLastHallucination / 30));
        }
        lastHallucinationStartPoint = [hallucinationX, hallucinationY];

        const effectiveScaleFactor = currentScaleFactor();
        hallucinationGraphics.stroke(hallucinationLineColor);
        for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
          const baseIndex = sampleIndex * 5;
          const hallucinationDX = data[baseIndex + 0] * effectiveScaleFactor;
          const hallucinationDY = data[baseIndex + 1] * effectiveScaleFactor;

          if (data[baseIndex + 4]) { // Corresponds to the logit for "end of drawing"
            break;
          }

          // We'll fade the line out over the last few samples.
          const alpha = p.lerp(100, 0, (sampleIndex - (hallucinatedSampleCount - 15)) / 15)
          const currentColor = p.color(hallucinationLineColor);
          currentColor.setAlpha(alpha);
          hallucinationGraphics.strokeWeight(lineWidth);
          hallucinationGraphics.stroke(currentColor);
          
          const isContinuingStroke = strokes[strokes.length - 1][2] === 0; // Look at the last pen index.
          if (sampleIndex > 0 && data[baseIndex - 5 + 2] || (sampleIndex === 0 && isContinuingStroke)) {
            hallucinationGraphics.line(hallucinationX, hallucinationY, hallucinationX + hallucinationDX, hallucinationY + hallucinationDY);
            if (data[baseIndex + 3]) {
              hallucinationGraphics.fill(0, 0, 255, 255);
              hallucinationGraphics.strokeWeight(1);
              hallucinationGraphics.ellipse(hallucinationX + hallucinationDX, hallucinationY + hallucinationDY, 5, 5);                
            }
          } else if ((sampleIndex > 0 && data[baseIndex - 5 + 3] && data[baseIndex + 2]) || (sampleIndex === 0 && !isContinuingStroke)) {
            hallucinationGraphics.fill(currentColor);
            hallucinationGraphics.ellipse(hallucinationX + hallucinationDX, hallucinationY + hallucinationDY, 3, 3);
          }

          hallucinationX += hallucinationDX;
          hallucinationY += hallucinationDY;
        }
      });
  }

  p.draw = function () {
    const mouseState = {
      x: p.mouseX,
      y: p.mouseY,
      down: p.mouseIsPressed
    };

    // record pen drawing from user:
    if (mouseState.down && (mouseState.x > 0) && mouseState.y < (screenHeight - 90)) { // pen is touching the paper
      if (lastMouseState === null) { // first time anything is written
        startingMouseState = mouseState;
        lastMouseState = mouseState;
        document.getElementById("hint").style.opacity = 0;
      }

      // Have we moved far enough to bother drawing anything?
      const dx = mouseState.x - lastMouseState.x;
      const dy = mouseState.y - lastMouseState.y;
      if (dx * dx + dy * dy > epsilon * epsilon) {
        if (lastMouseState.down) {
          drawingGraphics.stroke(lineColor);
          drawingGraphics.strokeWeight(lineWidth);
          drawingGraphics.line(lastMouseState.x, lastMouseState.y, lastMouseState.x + dx, lastMouseState.y + dy); // draw line connecting prev point to current point.
        }

        pendingRawLine.push([mouseState.x, mouseState.y]);
        updateModelStateUsingCurrentStrokes(false);
        lastMouseState = mouseState;
      }
    } else if (lastMouseState !== null) { // pen is above the paper
      updateModelStateUsingCurrentStrokes(true);
      pendingRawLine = [];
      pendingStrokeIndex = null;
      lastMouseState = mouseState;
    }

    if (currentModelState) {
      hallucinate(currentModelState, strokes);
    }

    const canvasScaleFactor = p.pixelDensity();
    p.image(hallucinationGraphics, 0, 0, screenWidth, screenHeight, 0, 0, screenWidth * canvasScaleFactor, screenHeight * canvasScaleFactor);
    p.image(drawingGraphics, 0, 0, screenWidth, screenHeight, 0, 0, screenWidth * canvasScaleFactor, screenHeight * canvasScaleFactor);
  };

  const onModelSelection = function (event) {
    const c = event.target.value;
    const modelMode = "gen";
    console.log("user wants to change to model " + c);
    setIsLoading(true);
    const callback = function (newModel) {
      setIsLoading(false);
      model = newModel;
      restart();
      clearScreen();
    }
    ModelImporter.change_model(model, c, modelMode, callback);
  };

  const onClear = function () {
    restart();
    clearScreen();
  };
};
const p5Instance = new p5(sketch, 'sketch');
