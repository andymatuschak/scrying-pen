// Based on an original demo at https://github.com/tensorflow/magenta-demos/tree/master/sketch-rnn-js.
// See LICENSE for full attribution and license details.

dl = deeplearn;
math = dl.ENV.math;

/**
 * Location of JSON models used for sketch-rnn-js
 */
var SketchRNNConfig = {
  BaseURL: "https://storage.googleapis.com/scrying-pen-models/"
};

/**
 * Tool to load simplify lines of a sketch using RDP Algorithm
 */
var DataTool = {};

(function (global) {
  "use strict";

  function simplify_line(V, tolerance) {
    // from https://gist.github.com/adammiller/826148
    // V ... [[x1,y1],[x2,y2],...] polyline
    // tol  ... approximation tolerance
    // ==============================================
    // Copyright 2002, softSurfer (www.softsurfer.com)
    // This code may be freely used and modified for any purpose
    // providing that this copyright notice is included with it.
    // SoftSurfer makes no warranty for this code, and cannot be held
    // liable for any real or imagined damage resulting from its use.
    // Users of this code must verify correctness for their application.
    // http://softsurfer.com/Archive/algorithm_0205/algorithm_0205.htm

    var tol=2.0;
    if (typeof(tolerance) === "number") {
      tol = tolerance;
    }

    var sum = function(u,v) {return [u[0]+v[0], u[1]+v[1]];}
    var diff = function(u,v) {return [u[0]-v[0], u[1]-v[1]];}
    var prod = function(u,v) {return [u[0]*v[0], u[1]*v[1]];}
    var dot = function(u,v) {return u[0]*v[0] + u[1]*v[1];}
    var norm2 = function(v) {return v[0]*v[0] + v[1]*v[1];}
    var norm = function(v) {return Math.sqrt(norm2(v));}
    var d2 = function(u,v) {return norm2(diff(u,v));}
    var d = function(u,v) {return norm(diff(u,v));}

    var simplifyDP = function( tol, v, j, k, mk ) {
      //  This is the Douglas-Peucker recursive simplification routine
      //  It just marks vertices that are part of the simplified polyline
      //  for approximating the polyline subchain v[j] to v[k].
      //  mk[] ... array of markers matching vertex array v[]
      if (k <= j+1) { // there is nothing to simplify
        return;
      }
      // check for adequate approximation by segment S from v[j] to v[k]
      var maxi = j;          // index of vertex farthest from S
      var maxd2 = 0;         // distance squared of farthest vertex
      var tol2 = tol * tol;  // tolerance squared
      var S = [v[j], v[k]];  // segment from v[j] to v[k]
      var u = diff(S[1], S[0]);   // segment direction vector
      var cu = norm2(u,u);     // segment length squared
      // test each vertex v[i] for max distance from S
      // compute using the Feb 2001 Algorithm's dist_Point_to_Segment()
      // Note: this works in any dimension (2D, 3D, ...)
      var  w;           // vector
      var Pb;                // point, base of perpendicular from v[i] to S
      var b, cw, dv2;        // dv2 = distance v[i] to S squared
      for (var i=j+1; i<k; i++) {
        // compute distance squared
        w = diff(v[i], S[0]);
        cw = dot(w,u);
        if ( cw <= 0 ) {
          dv2 = d2(v[i], S[0]);
        } else if ( cu <= cw ) {
          dv2 = d2(v[i], S[1]);
        } else {
          b = cw / cu;
          Pb = [S[0][0]+b*u[0], S[0][1]+b*u[1]];
          dv2 = d2(v[i], Pb);
        }
        // test with current max distance squared
        if (dv2 <= maxd2) {
          continue;
        }
        // v[i] is a new max vertex
        maxi = i;
        maxd2 = dv2;
      }
      if (maxd2 > tol2) {      // error is worse than the tolerance
        // split the polyline at the farthest vertex from S
        mk[maxi] = 1;      // mark v[maxi] for the simplified polyline
        // recursively simplify the two subpolylines at v[maxi]
        simplifyDP( tol, v, j, maxi, mk );  // polyline v[j] to v[maxi]
        simplifyDP( tol, v, maxi, k, mk );  // polyline v[maxi] to v[k]
      }
      // else the approximation is OK, so ignore intermediate vertices
      return;
    }

    var n = V.length;
    var sV = [];
    var i, k, m, pv;               // misc counters
    var tol2 = tol * tol;          // tolerance squared
    var vt = [];                       // vertex buffer, points
    var mk = [];                       // marker buffer, ints

    // STAGE 1.  Vertex Reduction within tolerance of prior vertex cluster
    vt[0] = V[0];              // start at the beginning
    for (i=k=1, pv=0; i<n; i++) {
      if (d2(V[i], V[pv]) < tol2) {
        continue;
      }
      vt[k++] = V[i];
      pv = i;
    }
    if (pv < n-1) {
      vt[k++] = V[n-1];      // finish at the end
    }

    // STAGE 2.  Douglas-Peucker polyline simplification
    mk[0] = mk[k-1] = 1;       // mark the first and last vertices
    simplifyDP( tol, vt, 0, k-1, mk );

    // copy marked vertices to the output simplified polyline
    for (i=m=0; i<k; i++) {
      if (mk[i]) {
        sV[m++] = vt[i];
      }
    }
    return sV;
  }

  /**
   * Clean wrapper method to use RDP function.
   */
  function simplify_lines(lines) {
    var result = [];
    var tolerance = 2.0;
    for (var i=0;i<lines.length;i++) {
      result.push(simplify_line(lines[i], tolerance));
    }
    return result;
  };

  /**
   * convert from stroke-5 format to polylines
   */
  function line_to_stroke(line, last_point, is_finished) {
    var penIndex;
    var stroke = [];
    var len;
    var p;
    var dx, dy;
    var x, y;
    var px, py;
    var j;
    px = last_point[0];
    py = last_point[1];
    len = line.length;
    if (len > 1) {
      for (j=0;j<len;j++) {
        p = line[j];
        //x = p.x;
        //y = p.y;
        x = p[0];
        y = p[1];
        if (j === len-1 && is_finished) {
          penIndex = 1;
        } else {
          penIndex = 0;
        }
        dx = x - px;
        dy = y - py;
        px = x;
        py = y;
        stroke.push([dx, dy, penIndex]);
      }
    }

    return stroke;

  };

  global.simplify_line = simplify_line;
  global.simplify_lines = simplify_lines;
  global.line_to_stroke = line_to_stroke;

})(DataTool);
(function(lib) {
  "use strict";
  if (typeof module === "undefined" || typeof module.exports === "undefined") {
    // usable in browser
  } else {
    module.exports = lib; // in nodejs
  }
})(DataTool);


/**
 * Simple tool to load JSON model files dynamically
 */
var ModelImporter = {};

(function(global) {
  "use strict";

  /**
   * load a given JSON model file dynamically
   */
  function loadJSON(filename, callback) {
    var xobj = new XMLHttpRequest();
        xobj.overrideMimeType("application/json");
        console.log(`Requesting ${filename}`);
        xobj.open('GET', filename, true);
        // Replace 'my_data' with the path to your file
        xobj.onreadystatechange = function () {
          if (xobj.readyState == 4 && xobj.status == "200") {
            // Required use of an anonymous callback
            // as .open() will NOT return a value but simply returns undefined in asynchronous mode
            callback(xobj.responseText);
          }
    };
    xobj.send(null);
  }

  // settings
  var init_model_data;
  var model_data_archive = [];
  var model_url = SketchRNNConfig.BaseURL;

  /**
   * assume given a parsed model model_raw_data, put it inside the model archives.
   * must do this at the beginning.
   */
  var set_init_model = function(model_raw_data) {
    init_model_data = JSON.parse(model_raw_data);
    model_data_archive[init_model_data[0].name+"_"+init_model_data[0].mode] = model_raw_data;
  };

  /**
   * Return's the current model selected in the database of possible models
   */
  var get_model_data = function() {
    return init_model_data;
  };

  /**
   * Have capability to override where to download the models from (if not on googlecloud)
   */
  var set_model_url = function(url) {
    model_url = url;
  };

  /**
   * Change the model to another class (ie, from ant to frog).
   */
  var change_model = function(model, class_name, class_type, call_back, do_not_cache) {
    if (model && typeof(class_name) === "string" && typeof(class_type) === "string") {
      var model_name = class_name + "." + class_type;
      console.log("attempting to load "+model_name);
      if (model_data_archive[model_name]) {
        console.log("changing with cached "+model_name);
        var new_model = new SketchRNN(JSON.parse(model_data_archive[model_name]));
        if (call_back) {
          call_back(new_model);
        }
        return;
      } else { // not cached
        var cache_model_mode = true;
        if (typeof do_not_cache === "undefined") {
          cache_model_mode = true;
        } else {
          if (do_not_cache) {
            cache_model_mode = false;
          } else {
            cache_model_mode = true;
          }
        }
        console.log("loading "+model_name+" dynamically");
        loadJSON(model_url+model_name+'.json', function(response) {
          console.log("callback from json load");
          // Parse JSON string into object
          var result = JSON.parse(response);
          if (cache_model_mode) {
            console.log("caching the model.");
            model_data_archive[model_name] = response; // cache it
          } else {
            console.log("not caching the model.");
          }
          var new_model = new SketchRNN(result);
          // model.load_model(model_data_archive[model_name]);
          if (call_back) {
            call_back(new_model);
          }
          return;
         });
      }
    }
  };

  global.get_model_data = get_model_data;
  global.set_init_model = set_init_model;
  global.set_model_url = set_model_url;
  global.change_model = change_model;

})(ModelImporter);
(function(lib) {
  "use strict";
  if (typeof module === "undefined" || typeof module.exports === "undefined") {
    //window.jsfeat = lib; // in ordinary browser attach library to window
  } else {
    module.exports = lib; // in nodejs
  }
})(ModelImporter);

/**
 * Internal LSTM class used by sketch-rnn
 * @class
 * @constructor
 */
function LSTMCell(num_units, input_size, Wxh, Whh, bias) {
  this.num_units = num_units;
  this.input_size = input_size;
  this.Wxh = Wxh;
  this.Whh = Whh;
  this.bias = bias;
  this.forget_bias = 1.0;
  this.Wfull= math.switchDim(math.concat2D(math.switchDim(Wxh, [1, 0]), math.switchDim(Whh, [1, 0]), 1), [1, 0]);
}
LSTMCell.prototype.zero_state = function() {
  return [dl.NDArray.zeros([this.num_units]), dl.NDArray.zeros([this.num_units])];
};
LSTMCell.prototype.forward = function(x, h, c) {
  return math.scope((keep, track) => {

    var concat = math.concat1D(x, h);
    var hidden = math.add(math.vectorTimesMatrix(concat, this.Wfull), this.bias);
    var num_units = this.num_units;
    var forget_bias = this.forget_bias;

    var i=math.sigmoid(math.slice1D(hidden, 0*num_units, num_units));
    var g=math.tanh(math.slice1D(hidden, 1*num_units, num_units));
    var f=math.sigmoid(math.add(math.slice1D(hidden, 2*num_units, num_units), track(dl.Scalar.new(forget_bias))));
    var o=math.sigmoid(math.slice1D(hidden, 3*num_units, num_units));

    var new_c = math.add(math.multiply(c, f), math.multiply(g, i));
    var new_h = math.multiply(math.tanh(new_c), o);

    return [new_h, new_c];
  });
};
LSTMCell.prototype.encode = function(sequence) {
  var x;
  var state = this.zero_state();
  var h = state[0];
  var c = state[1];
  var N = sequence.length;
  for (var i=0;i<N;i++) {
    x = nj.array(sequence[i]);
    state = this.forward(x, h, c);
    h = state[0];
    c = state[1];
  }
  return h;
};

/**
 * The sketch-rnn model. Please see README.md for documentation.
 * @class
 */
function SketchRNN(model_raw_data) {
  "use strict";

  // settings
  var info;
  var dimensions;
  var num_blobs;
  var weights;
  var max_weight;
  var N_mixture;

  var max_seq_len;

  // model variables:
  var enc_fw_lstm_W_xh, enc_fw_lstm_W_hh, enc_fw_lstm_bias, enc_bw_lstm_W_xh,enc_bw_lstm_W_hh,enc_bw_lstm_bias,enc_mu_w,enc_mu_b,enc_sigma_w,enc_sigma_b,enc_w,enc_b,dec_output_w,dec_output_b,dec_lstm_W_xh,dec_lstm_W_hh,dec_lstm_bias;
  var dec_num_units, dec_input_size, enc_num_units, enc_input_size, z_size;
  var enc_fw_lstm, enc_bw_lstm, dec_lstm;

  /**
   * deals with decompressing b64 models to float arrays.
   */
  function string_to_uint8array(b64encoded) {
    var u8 = new Uint8Array(atob(b64encoded).split("").map(function(c) {
      return c.charCodeAt(0); }));
    return u8;
  }
  function uintarray_to_string(u8) {
    var s = "";
    for (var i = 0, len = u8.length; i < len; i++) {
      s += String.fromCharCode(u8[i]);
    }
    var b64encoded = btoa(s);
    return b64encoded;
  };
  function string_to_array(s) {
    var u = string_to_uint8array(s);
    var result = new Int16Array(u.buffer);
    return result;
  };
  function array_to_string(a) {
    var u = new Uint8Array(a.buffer);
    var result = uintarray_to_string(u);
    return result;
  };

  // Random numbers util (from https://github.com/karpathy/recurrentjs)
  var return_v = false;
  var v_val = 0.0;
  function gaussRandom() {
    if(return_v) {
      return_v = false;
      return v_val;
    }
    var u = 2*Math.random()-1;
    var v = 2*Math.random()-1;
    var r = u*u + v*v;
    if(r == 0 || r > 1) return gaussRandom();
    var c = Math.sqrt(-2*Math.log(r)/r);
    v_val = v*c; // cache this
    return_v = true;
    return u*c;
  }
  function randf(a, b) { return Math.random()*(b-a)+a; };
  // from http://www.math.grin.edu/~mooret/courses/math336/bivariate-normal.html
  function birandn(weights, z1, z2, mu1, mu2, std1, std2, rho) {
    var ones = dl.NDArray.zerosLike(rho);
    ones.fill(1);

    var x = math.add(
      math.add(
        math.elementWiseMul(
          math.sqrt(math.sub(ones, math.elementWiseMul(rho, rho))),
          math.elementWiseMul(std1, z1)
        ),
        math.elementWiseMul(rho, math.elementWiseMul(std1, z2))
      ),
      mu1
    );
    var y = math.add(
      math.elementWiseMul(std2, z2),
      mu2
    );
    return [math.dotProduct(weights, x), math.dotProduct(weights, y)];
  };

  /**
   * loads a JSON-parsed model in its specified format.
   */
  function load_model(model_raw_data) {
    "use strict";
    var i, j;

    info = model_raw_data[0];
    dimensions = model_raw_data[1];
    num_blobs = dimensions.length;
    var weightsIn = model_raw_data[2];
    weights = Array(weightsIn.length)
    max_weight = 10.0;
    N_mixture=20;

    max_seq_len = info.max_seq_len;

    for (i=0;i<num_blobs;i++) {
      weights[i] = dl.Array1D.new(new Float32Array(string_to_array(weightsIn[i])), 'float32');
      weights[i] = math.divide(weights[i], dl.Scalar.new(32767));
      weights[i] = math.multiply(weights[i], dl.Scalar.new(max_weight));
      if(dimensions[i].length == 2) {
        var d = dimensions[i];
        var d1 = d[0], d2 = d[1];
        weights[i] = weights[i].reshape([d1, d2]);
      }
    }

    if(info.mode === 2 || info.mode === "gen") { // 0 or 1 - vae, 2 - gen
      dec_output_w = weights[0];
      dec_output_b = weights[1];
      dec_lstm_W_xh = weights[2];
      dec_lstm_W_hh = weights[3];
      dec_lstm_bias = weights[4];
    } else {
      enc_fw_lstm_W_xh = weights[0];
      enc_fw_lstm_W_hh = weights[1];
      enc_fw_lstm_bias = weights[2];
      enc_bw_lstm_W_xh = weights[3];
      enc_bw_lstm_W_hh = weights[4];
      enc_bw_lstm_bias = weights[5];
      enc_mu_w = weights[6];
      enc_mu_b = weights[7];
      enc_sigma_w = weights[8];
      enc_sigma_b = weights[9];
      enc_w = weights[10];
      enc_b = weights[11];
      dec_output_w = weights[12];
      dec_output_b = weights[13];
      dec_lstm_W_xh = weights[14];
      dec_lstm_W_hh = weights[15];
      dec_lstm_bias = weights[16];
      enc_num_units = enc_fw_lstm_W_hh.shape[0];
      enc_input_size = enc_fw_lstm_W_xh.shape[0];
      z_size = enc_w.shape[0];

      enc_fw_lstm = new LSTMCell(enc_num_units, enc_input_size, enc_fw_lstm_W_xh, enc_fw_lstm_W_hh, enc_fw_lstm_bias);
      enc_bw_lstm = new LSTMCell(enc_num_units, enc_input_size, enc_bw_lstm_W_xh, enc_bw_lstm_W_hh, enc_bw_lstm_bias);
    }

    dec_num_units = dec_lstm_W_hh.shape[0];
    dec_input_size = dec_lstm_W_xh.shape[0];

    dec_lstm = new LSTMCell(dec_num_units, dec_input_size, dec_lstm_W_xh, dec_lstm_W_hh, dec_lstm_bias);
    console.log("loading model...");
    console.log("class="+info.name);
    console.log("version="+info.version);
    console.log("model type="+info.mode);
    console.log("train size="+info.name);
    console.log("scale factor="+Math.round(1000*info.scale_factor)/1000);
    console.log("reconst loss="+Math.round(1000*info.r_loss)/1000);
    console.log("kl loss="+Math.round(1000*info.kl_loss)/1000);
    console.log("max seq len="+info.max_seq_len);
    console.log("training sample size="+info.training_size);
  };

  /**
   * return an empty state of sketch-rnn
   */
  function zero_state() {
    return dec_lstm.zero_state();
  };

  /**
   * returns a copy of the rnn state (for multiple predictions given the same starting point)
   */
  function copy_state(state) {
    var h = math.clone(state[0]);
    var c = math.clone(state[1]);
    return [h, c];
  };

  /**
   * Initially the input is zero for "gen" type models
   */
  function zero_input() {
    return [0, 0, 0];
  };

  /**
   * update the rnn with input x, state s, and optional latent vector y.
   */
  function update(x, s, y) {
    return math.scope((keep, track) => {
      // y is an optional vector parameter, used for conditional mode only.
      var x_ = null;
      if (x instanceof dl.Array1D) {
        x_ = x;
      } else {
        // HACK: beware, the fact that we divide by scale_factor here but not in the other branch is pretty tricksy
        x_ = track(dl.Array1D.new([x[0], x[1], x[2] === 0 ? 1 : 0, x[2] === 1 ? 1 : 0, x[2] === 2 ? 1 : 0]));
      }

      var lstm_input, rnn_state;

      if (y) {
        var z = nj.array(y);
        lstm_input = nj.concatenate([x_, z]);
      } else {
        lstm_input = x_;
      }

      rnn_state = dec_lstm.forward(lstm_input, s[0], s[1]);

      return rnn_state;
    });
  };

  /**
   * Gets the parameters of the mixture density distribution for the next point
   */
  function get_pdf(s) {
    var h = s[0];
    var NOUT = N_mixture;
    var z=math.add(math.vectorTimesMatrix(h, dec_output_w), dec_output_b);
    var z_pen_logits = math.slice1D(z, 0, 3);
    var z_pi = math.slice1D(z, 3+NOUT*0, NOUT);
    var z_mu1 = math.slice1D(z, 3+NOUT*1, NOUT);
    var z_mu2 = math.slice1D(z, 3+NOUT*2, NOUT);
    var z_sigma1 = math.exp(math.slice1D(z, 3+NOUT*3, NOUT));
    var z_sigma2 = math.exp(math.slice1D(z, 3+NOUT*4, NOUT));
    var z_corr = math.tanh(math.slice1D(z, 3+NOUT*5, NOUT));
    z_pen_logits = math.sub(z_pen_logits, math.max(z_pen_logits));
    var z_pen = math.softmax(z_pen_logits);
    z_pi = math.sub(z_pi, math.max(z_pi));
    z_pi = math.softmax(z_pi);

    return [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen];
  };

  /**
   * sample from a categorial distribution
   */
  function sample_softmax(z_sample) {
    return math.multinomial(z_sample, 1).asScalar();
  };

  /**
   * adjust the temperature of a categorial distribution
   */
  function adjust_temp(z_old, temp) {
    return math.scope((keep, track) => {
      var z = math.clone(z_old);
      var i;
      var x;
      z = math.divide(math.log(z), track(dl.Scalar.new(temp)))
      x = math.max(z);
      z = math.sub(z, x);
      z = math.exp(z);
      x = math.sum(z);
      z = math.divide(z, x);
      return z;
    });
  };

  /**
   * samples the next point of the sketch given pdf parameters and optional temperature params
   */
  function sample(z, temperature, softmax_temperature) {
    return math.scope((keep, track) => {
      // z is [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen]
      // returns [x, y, eos]
      // optional softmax_temperature
      var temp=0.65;
      if (typeof(temperature) === "number") {
        temp = temperature;
      }
      var softmax_temp = 0.5+temp*0.5;
      if (typeof(softmax_temperature) === "number") {
        softmax_temp = softmax_temperature;
      }
      var z_0 = adjust_temp(z[0], softmax_temp);
      var z_6 = adjust_temp(z[6], softmax_temp);

      var pen_idx = sample_softmax(z_6);

      var tempScalar = track(dl.Scalar.new(Math.sqrt(temp)));
      var delta = birandn(
        z[0],
        dl.Array1D.randNormal(z[0].shape),
        dl.Array1D.randNormal(z[0].shape),
        z[1],
        z[2],
        math.multiply(tempScalar, z[3]),
        math.multiply(tempScalar, z[4]),
        z[5]
      );

      return math.concat1D(
        math.concat1D(delta[0].as1D(), delta[1].as1D()),
        math.oneHot(pen_idx.as1D(), 3).as1D()
      )
    })
  }

  load_model(model_raw_data);

  function get_info() {
    return this.info;
  };

  this.zero_state = zero_state;
  this.zero_input = zero_input;
  this.copy_state = copy_state;
  this.update = update;
  this.get_pdf = get_pdf;
  this.sample = sample;
  this.info = info;
  this.name = info.name;
  this.mode = info.mode;
  this.get_info = get_info;
}
