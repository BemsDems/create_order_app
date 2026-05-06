/* TCN multi-horizon inference — pure browser JS port of the original numpy
 * backend. No external libs. Exposes globals on window.TCN.
 *
 * Pipeline mirrors stock-forecast-backend/app/{features,inference}.py.
 */
(function () {
  'use strict';

  const TCN = {};

  // ---- Numerics --------------------------------------------------------
  const NaNv = NaN;

  function arrFill(n, v) { const a = new Float64Array(n); if (v !== 0) a.fill(v); return a; }
  function isFiniteNum(x) { return Number.isFinite(x); }

  function rollingMean(x, n) {
    const N = x.length;
    const out = new Float64Array(N); out.fill(NaNv);
    if (n <= 0 || N < n) return out;
    let sum = 0, count = 0;
    for (let i = 0; i < n; i++) {
      const v = x[i];
      if (Number.isFinite(v)) { sum += v; count++; }
    }
    if (count >= n) out[n - 1] = sum / n;
    for (let i = n; i < N; i++) {
      const add = x[i], drop = x[i - n];
      if (Number.isFinite(add)) { sum += add; count++; }
      if (Number.isFinite(drop)) { sum -= drop; count--; }
      if (count >= n) out[i] = sum / n;
    }
    return out;
  }

  function rollingStd(x, n, ddof) {
    if (ddof === undefined) ddof = 1;
    const N = x.length;
    const out = new Float64Array(N); out.fill(NaNv);
    if (n <= 0 || n - ddof <= 0) return out;
    for (let i = n - 1; i < N; i++) {
      let nan = false, mean = 0;
      for (let j = i - n + 1; j <= i; j++) {
        if (!Number.isFinite(x[j])) { nan = true; break; }
        mean += x[j];
      }
      if (nan) continue;
      mean /= n;
      let sq = 0;
      for (let j = i - n + 1; j <= i; j++) sq += (x[j] - mean) * (x[j] - mean);
      out[i] = Math.sqrt(sq / (n - ddof));
    }
    return out;
  }

  function rollingMax(x, n) {
    const N = x.length;
    const out = new Float64Array(N); out.fill(NaNv);
    for (let i = n - 1; i < N; i++) {
      let m = -Infinity, nan = false;
      for (let j = i - n + 1; j <= i; j++) {
        if (!Number.isFinite(x[j])) { nan = true; break; }
        if (x[j] > m) m = x[j];
      }
      if (!nan) out[i] = m;
    }
    return out;
  }
  function rollingMin(x, n) {
    const N = x.length;
    const out = new Float64Array(N); out.fill(NaNv);
    for (let i = n - 1; i < N; i++) {
      let m = Infinity, nan = false;
      for (let j = i - n + 1; j <= i; j++) {
        if (!Number.isFinite(x[j])) { nan = true; break; }
        if (x[j] < m) m = x[j];
      }
      if (!nan) out[i] = m;
    }
    return out;
  }

  function ffill(x) {
    const N = x.length, out = new Float64Array(N);
    let last = NaNv;
    for (let i = 0; i < N; i++) {
      out[i] = Number.isFinite(x[i]) ? (last = x[i]) : last;
    }
    return out;
  }

  function logretClipped(close, lag) {
    const N = close.length;
    const out = new Float64Array(N); out.fill(NaNv);
    if (lag <= 0 || lag >= N) return out;
    for (let i = lag; i < N; i++) {
      const denom = close[i - lag];
      if (!Number.isFinite(denom) || denom === 0 || !Number.isFinite(close[i])) continue;
      let r = close[i] / denom;
      if (r < 0.5) r = 0.5;
      if (r > 2.0) r = 2.0;
      out[i] = Math.log(r);
    }
    return out;
  }

  function rsi(close, period) {
    if (period === undefined) period = 14;
    const N = close.length;
    const gain = new Float64Array(N); gain.fill(NaNv);
    const loss = new Float64Array(N); loss.fill(NaNv);
    for (let i = 1; i < N; i++) {
      const d = close[i] - close[i - 1];
      if (Number.isFinite(d)) {
        gain[i] = d > 0 ? d : 0;
        loss[i] = d < 0 ? -d : 0;
      }
    }
    const avgG = rollingMean(gain, period);
    const avgL = rollingMean(loss, period);
    const out = new Float64Array(N); out.fill(NaNv);
    for (let i = 0; i < N; i++) {
      if (Number.isFinite(avgG[i]) && Number.isFinite(avgL[i])) {
        const rs = avgG[i] / (avgL[i] + 1e-12);
        out[i] = 100 - 100 / (1 + rs);
      }
    }
    return out;
  }

  // dates as ms-since-epoch numbers (UTC midnight). All series sorted ascending.
  function mergeBackward(targetDates, srcDates, srcValues) {
    const out = new Float64Array(targetDates.length); out.fill(NaNv);
    if (srcDates.length === 0) return out;
    let j = 0;
    for (let i = 0; i < targetDates.length; i++) {
      const d = targetDates[i];
      while (j < srcDates.length && srcDates[j] <= d) j++;
      // j is the first index strictly greater; previous (j-1) is what we want
      if (j > 0) out[i] = srcValues[j - 1];
    }
    return out;
  }

  const ONE_DAY_MS = 86400000;

  function ttmDivYield(targetDates, targetClose, divDates, divValues, lagDays) {
    if (lagDays === undefined) lagDays = 1;
    const n = targetDates.length;
    const divYield = new Float64Array(n);
    const daysSince = new Float64Array(n); daysSince.fill(1.0);
    const isMissing = new Float64Array(n); isMissing.fill(1.0);

    if (!divDates || divDates.length === 0) return [divYield, daysSince, isMissing];

    // eff = registry_close + lagDays
    const eff = new Float64Array(divDates.length);
    for (let i = 0; i < divDates.length; i++) eff[i] = divDates[i] + lagDays * ONE_DAY_MS;

    const oneYear = 365 * ONE_DAY_MS;

    let lastIdx = -1;
    for (let i = 0; i < n; i++) {
      const d = targetDates[i];
      while (lastIdx + 1 < eff.length && eff[lastIdx + 1] <= d) lastIdx++;
      // sum of divValues with eff in (d - 365d, d]
      const lo = d - oneYear;
      let ttm = 0;
      for (let k = 0; k <= lastIdx; k++) {
        if (eff[k] > lo && eff[k] <= d) ttm += divValues[k];
      }
      const cl = targetClose[i];
      if (cl > 0 && ttm > 0) {
        let dy = ttm / cl;
        if (dy < 0) dy = 0;
        if (dy > 0.30) dy = 0.30;
        divYield[i] = dy;
      }
      if (lastIdx >= 0) {
        let delta = Math.floor((d - eff[lastIdx]) / ONE_DAY_MS);
        if (delta < 0) delta = 0;
        if (delta > 365) delta = 365;
        daysSince[i] = delta / 365.0;
        isMissing[i] = 0.0;
      }
    }
    return [divYield, daysSince, isMissing];
  }

  // ---- Build features (28 columns) -------------------------------------
  function buildFeatures(opts) {
    const dates = opts.dates;
    const close = Float64Array.from(opts.close);
    const high = Float64Array.from(opts.high);
    const low = Float64Array.from(opts.low);
    const volume = Float64Array.from(opts.volume);
    const N = close.length;

    const logret_1 = logretClipped(close, 1);
    const logret_2 = logretClipped(close, 2);
    const logret_3 = logretClipped(close, 3);
    const logret_5 = logretClipped(close, 5);
    const logret_10 = logretClipped(close, 10);

    const sma_20 = rollingMean(close, 20);
    const sma_200 = rollingMean(close, 200);

    const trend_up_20 = new Float64Array(N); trend_up_20.fill(NaNv);
    for (let i = 0; i < N; i++)
      if (Number.isFinite(sma_20[i])) trend_up_20[i] = close[i] > sma_20[i] ? 1.0 : 0.0;

    const sma_200_ff = ffill(sma_200);
    const trend_up_200 = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      if (Number.isFinite(sma_200_ff[i])) trend_up_200[i] = close[i] > sma_200_ff[i] ? 1.0 : 0.0;
      else trend_up_200[i] = 0.0;
    }

    const vol_ma_20 = rollingMean(volume, 20);
    const vol_rel = new Float64Array(N);
    const vol_spike = new Float64Array(N); vol_spike.fill(NaNv);
    for (let i = 0; i < N; i++) {
      if (Number.isFinite(vol_ma_20[i])) {
        let r = volume[i] / (vol_ma_20[i] + 1e-9);
        if (r < 0.1) r = 0.1; if (r > 3.0) r = 3.0;
        vol_rel[i] = r;
        vol_spike[i] = r > 2.0 ? 1.0 : 0.0;
      } else {
        vol_rel[i] = volume[i] / 1e-9; // mirror python behaviour, will be clipped
        if (vol_rel[i] < 0.1) vol_rel[i] = 0.1;
        if (vol_rel[i] > 3.0) vol_rel[i] = 3.0;
      }
    }

    const rsi_14 = rsi(close, 14);
    for (let i = 0; i < N; i++) {
      if (Number.isFinite(rsi_14[i])) {
        if (rsi_14[i] < 0) rsi_14[i] = 0;
        if (rsi_14[i] > 100) rsi_14[i] = 100;
      }
    }
    const rsi_oversold = new Float64Array(N); rsi_oversold.fill(NaNv);
    const rsi_overbought = new Float64Array(N); rsi_overbought.fill(NaNv);
    for (let i = 0; i < N; i++) {
      if (Number.isFinite(rsi_14[i])) {
        rsi_oversold[i] = rsi_14[i] < 30 ? 1.0 : 0.0;
        rsi_overbought[i] = rsi_14[i] > 70 ? 1.0 : 0.0;
      }
    }

    const high_20 = rollingMax(high, 20);
    const low_20 = rollingMin(low, 20);
    const price_pos_20 = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let v = (close[i] - low_20[i]) / (high_20[i] - low_20[i] + 1e-9);
      if (!Number.isFinite(v)) v = NaNv;
      else { if (v < 0) v = 0; if (v > 1) v = 1; }
      price_pos_20[i] = v;
    }

    const volatility_20 = rollingStd(logret_1, 20, 1);
    for (let i = 0; i < N; i++) {
      if (Number.isFinite(volatility_20[i])) {
        if (volatility_20[i] < 0) volatility_20[i] = 0;
        if (volatility_20[i] > 0.1) volatility_20[i] = 0.1;
      }
    }

    // USD/RUB macro
    let usdrub_logret_1, usdrub_logret_5, usdrub_vol_20;
    if (opts.usd && opts.usd.dates && opts.usd.dates.length > 0) {
      const usd_close = ffill(mergeBackward(dates, opts.usd.dates, opts.usd.values));
      usdrub_logret_1 = logretClipped(usd_close, 1);
      usdrub_logret_5 = logretClipped(usd_close, 5);
      const v = rollingStd(usdrub_logret_1, 20, 1);
      for (let i = 0; i < v.length; i++) {
        if (Number.isFinite(v[i])) { if (v[i] < 0) v[i] = 0; if (v[i] > 0.1) v[i] = 0.1; }
      }
      usdrub_vol_20 = v;
    } else {
      usdrub_logret_1 = new Float64Array(N);
      usdrub_logret_5 = new Float64Array(N);
      usdrub_vol_20 = new Float64Array(N);
    }

    // IMOEX macro
    let imoex_logret_1, imoex_logret_5, imoex_logret_20;
    if (opts.imoex && opts.imoex.dates && opts.imoex.dates.length > 0) {
      const imoex_close = ffill(mergeBackward(dates, opts.imoex.dates, opts.imoex.values));
      imoex_logret_1 = logretClipped(imoex_close, 1);
      imoex_logret_5 = logretClipped(imoex_close, 5);
      imoex_logret_20 = logretClipped(imoex_close, 20);
    } else {
      imoex_logret_1 = new Float64Array(N);
      imoex_logret_5 = new Float64Array(N);
      imoex_logret_20 = new Float64Array(N);
    }

    // Brent (zeros)
    const zeros = new Float64Array(N);

    // Dividends
    let div_y, days_since, div_missing;
    if (opts.div && opts.div.dates && opts.div.dates.length > 0) {
      const r = ttmDivYield(dates, close, opts.div.dates, opts.div.values);
      div_y = r[0]; days_since = r[1]; div_missing = r[2];
    } else {
      div_y = new Float64Array(N);
      days_since = new Float64Array(N); days_since.fill(1.0);
      div_missing = new Float64Array(N); div_missing.fill(1.0);
    }

    // Stack columns (N, 28)
    const cols = [
      logret_1, logret_2, logret_3, logret_5, logret_10,
      trend_up_20, trend_up_200,
      vol_rel, vol_spike,
      rsi_14, rsi_oversold, rsi_overbought,
      price_pos_20, volatility_20,
      div_y, days_since, div_missing,
      usdrub_logret_1, usdrub_logret_5, usdrub_vol_20,
      zeros, zeros,
      imoex_logret_1, imoex_logret_5, imoex_logret_20,
      zeros, zeros, zeros,
    ];

    // critical = [logret_1, logret_10, sma_20, vol_ma_20, rsi_14, price_pos_20]
    const matrix = [];
    const keptDates = [];
    for (let i = 0; i < N; i++) {
      if (
        Number.isFinite(logret_1[i]) && Number.isFinite(logret_10[i]) &&
        Number.isFinite(sma_20[i]) && Number.isFinite(vol_ma_20[i]) &&
        Number.isFinite(rsi_14[i]) && Number.isFinite(price_pos_20[i])
      ) {
        const row = new Float32Array(28);
        for (let c = 0; c < 28; c++) {
          const v = cols[c][i];
          row[c] = Number.isFinite(v) ? v : 0;
        }
        matrix.push(row);
        keptDates.push(dates[i]);
      }
    }
    return { matrix: matrix, dates: keptDates };
  }

  // ---- TCN forward pass ------------------------------------------------
  function layerNorm(x, gamma, beta, eps) {
    if (eps === undefined) eps = 1e-3;
    const T = x.length;
    const F = x[0].length;
    const out = new Array(T);
    for (let t = 0; t < T; t++) {
      let mean = 0;
      for (let f = 0; f < F; f++) mean += x[t][f];
      mean /= F;
      let varSum = 0;
      for (let f = 0; f < F; f++) { const d = x[t][f] - mean; varSum += d * d; }
      const v = varSum / F;
      const inv = 1 / Math.sqrt(v + eps);
      const row = new Float32Array(F);
      for (let f = 0; f < F; f++) row[f] = (x[t][f] - mean) * inv * gamma[f] + beta[f];
      out[t] = row;
    }
    return out;
  }

  // Causal Conv1D matching Keras padding='causal'.
  // x: (T, F_in), kernel: (k, F_in, F_out) (3-D nested), bias: (F_out,)
  function causalConv1D(x, kernel, bias, dilation) {
    const T = x.length;
    const F_in = x[0].length;
    const k = kernel.length;
    const F_out = bias.length;
    const pad = (k - 1) * dilation;
    // xp = zeros(pad) ++ x
    const out = new Array(T);
    for (let t = 0; t < T; t++) {
      const o = new Float32Array(F_out);
      for (let f = 0; f < F_out; f++) o[f] = bias[f];
      out[t] = o;
    }
    for (let ki = 0; ki < k; ki++) {
      const offset = ki * dilation;     // index into xp
      const kk = kernel[ki];            // (F_in, F_out)
      // For each output position t: input row index in real x is (offset + t - pad)
      for (let t = 0; t < T; t++) {
        const srcIdx = offset + t - pad;
        if (srcIdx < 0) continue;       // padded zero, no contribution
        const row = x[srcIdx];
        const o = out[t];
        for (let fi = 0; fi < F_in; fi++) {
          const v = row[fi];
          if (v === 0) continue;
          const krow = kk[fi];
          for (let fo = 0; fo < F_out; fo++) o[fo] += v * krow[fo];
        }
      }
    }
    return out;
  }

  function relu(x) {
    for (let t = 0; t < x.length; t++) {
      const r = x[t];
      for (let f = 0; f < r.length; f++) if (r[f] < 0) r[f] = 0;
    }
    return x;
  }

  function batchNorm(x, gamma, beta, mm, mv, eps) {
    if (eps === undefined) eps = 1e-3;
    const F = x[0].length;
    const inv = new Float32Array(F);
    for (let f = 0; f < F; f++) inv[f] = gamma[f] / Math.sqrt(mv[f] + eps);
    for (let t = 0; t < x.length; t++) {
      const r = x[t];
      for (let f = 0; f < F; f++) r[f] = (r[f] - mm[f]) * inv[f] + beta[f];
    }
    return x;
  }

  function gap(x) {
    const T = x.length, F = x[0].length;
    const out = new Float32Array(F);
    for (let t = 0; t < T; t++) {
      const r = x[t];
      for (let f = 0; f < F; f++) out[f] += r[f];
    }
    for (let f = 0; f < F; f++) out[f] /= T;
    return out;
  }

  function denseRelu(x, kernel, bias) {
    // kernel: (F_in, F_out)
    const F_in = kernel.length;
    const F_out = bias.length;
    const out = new Float32Array(F_out);
    for (let fo = 0; fo < F_out; fo++) out[fo] = bias[fo];
    for (let fi = 0; fi < F_in; fi++) {
      const v = x[fi];
      if (v === 0) continue;
      const krow = kernel[fi];
      for (let fo = 0; fo < F_out; fo++) out[fo] += v * krow[fo];
    }
    for (let fo = 0; fo < F_out; fo++) if (out[fo] < 0) out[fo] = 0;
    return out;
  }

  function denseSigmoid(x, kernel, bias) {
    const F_in = kernel.length;
    let s = bias[0];
    for (let fi = 0; fi < F_in; fi++) s += x[fi] * kernel[fi][0];
    return 1 / (1 + Math.exp(-s));
  }

  function robustScale(features, center, scale) {
    // features: array of 28-element rows; center/scale: length-28
    const T = features.length;
    const out = new Array(T);
    for (let t = 0; t < T; t++) {
      const row = new Float32Array(28);
      for (let f = 0; f < 28; f++) row[f] = (features[t][f] - center[f]) / scale[f];
      out[t] = row;
    }
    return out;
  }

  // ---- Weight reshape helpers (JSON arrays → typed/nested) -------------
  function flatTo2D(flat, rows, cols) {
    const out = new Array(rows);
    for (let r = 0; r < rows; r++) {
      const row = new Float32Array(cols);
      for (let c = 0; c < cols; c++) row[c] = flat[r * cols + c];
      out[r] = row;
    }
    return out;
  }
  function flatTo3D(flat, k, fin, fout) {
    const out = new Array(k);
    for (let ki = 0; ki < k; ki++) {
      const m = new Array(fin);
      for (let fi = 0; fi < fin; fi++) {
        const row = new Float32Array(fout);
        const base = (ki * fin + fi) * fout;
        for (let fo = 0; fo < fout; fo++) row[fo] = flat[base + fo];
        m[fi] = row;
      }
      out[ki] = m;
    }
    return out;
  }

  function flatten(arr) {
    // recursively flatten nested JSON arrays
    const out = [];
    const stack = [arr];
    while (stack.length) {
      const top = stack.pop();
      if (Array.isArray(top)) {
        for (let i = top.length - 1; i >= 0; i--) stack.push(top[i]);
      } else {
        out.push(top);
      }
    }
    return out;
  }

  function prepareWeights(json) {
    const shapes = json._shapes;
    const W = {};
    for (const k of json._keys) {
      const shape = shapes[k];
      const flat = flatten(json[k]);
      if (shape.length === 1) {
        W[k] = Float32Array.from(flat);
      } else if (shape.length === 2) {
        W[k] = flatTo2D(flat, shape[0], shape[1]);
      } else if (shape.length === 3) {
        W[k] = flatTo3D(flat, shape[0], shape[1], shape[2]);
      } else {
        throw new Error('unexpected weight shape for ' + k + ': ' + shape);
      }
    }
    return W;
  }

  // ---- Top-level predict -----------------------------------------------
  TCN.SEQ_LEN = 30;
  TCN.N_FEATURES = 28;
  TCN.meta = null;
  TCN.weights = {}; // group → W

  TCN.loadMeta = async function (baseUrl) {
    const r = await fetch(baseUrl + '/tcn_meta.json', { cache: 'force-cache' });
    if (!r.ok) throw new Error('meta load failed: ' + r.status);
    TCN.meta = await r.json();
    return TCN.meta;
  };

  TCN.loadGroup = async function (group, baseUrl) {
    if (TCN.weights[group]) return TCN.weights[group];
    const r = await fetch(baseUrl + '/tcn_' + group + '_weights.json', { cache: 'force-cache' });
    if (!r.ok) throw new Error('weights load failed for ' + group + ': ' + r.status);
    const json = await r.json();
    TCN.weights[group] = prepareWeights(json);
    return TCN.weights[group];
  };

  TCN.horizonToGroup = function (h) {
    if (!TCN.meta) throw new Error('meta not loaded');
    const g = TCN.meta.horizon_to_group[String(h)];
    if (!g) throw new Error('unsupported horizon ' + h);
    return g;
  };

  TCN.predict = function (features, horizon) {
    if (!TCN.meta) throw new Error('meta not loaded');
    const group = TCN.horizonToGroup(horizon);
    const W = TCN.weights[group];
    if (!W) throw new Error('weights for ' + group + ' not loaded');
    if (features.length < TCN.SEQ_LEN) {
      throw new Error('need ' + TCN.SEQ_LEN + ' rows, got ' + features.length);
    }
    const window = features.slice(features.length - TCN.SEQ_LEN);

    const sc = TCN.meta.scalers[group];
    let x = robustScale(window, sc.center, sc.scale);

    x = layerNorm(x, W.ln_g, W.ln_b);
    x = causalConv1D(x, W.conv0_k, W.conv0_b, 1);
    x = relu(x);
    x = batchNorm(x, W.bn0_g, W.bn0_b, W.bn0_mm, W.bn0_mv);
    x = causalConv1D(x, W.conv1_k, W.conv1_b, 2);
    x = relu(x);
    x = batchNorm(x, W.bn1_g, W.bn1_b, W.bn1_mm, W.bn1_mv);
    x = causalConv1D(x, W.conv2_k, W.conv2_b, 4);
    x = relu(x);
    x = batchNorm(x, W.bn2_g, W.bn2_b, W.bn2_mm, W.bn2_mv);

    const pooled = gap(x);
    const h0 = denseRelu(pooled, W.d0_k, W.d0_b);
    const h1 = denseRelu(h0, W.d1_k, W.d1_b);
    const prob = denseSigmoid(h1, W.d2_k, W.d2_b);

    return {
      prob_up: prob,
      group: group,
      threshold: TCN.meta.horizon_groups[group].thr,
    };
  };

  // ---- MOEX data fetch -------------------------------------------------
  const MOEX_BASE = 'https://iss.moex.com/iss';

  function dateToMs(s) {
    // 'YYYY-MM-DD' → UTC midnight epoch ms
    const [y, m, d] = s.split('-').map(Number);
    return Date.UTC(y, m - 1, d);
  }

  async function fetchPaged(url, dataKey) {
    const out = [];
    let start = 0;
    while (true) {
      const sep = url.includes('?') ? '&' : '?';
      const r = await fetch(url + sep + 'start=' + start);
      if (!r.ok) throw new Error('MOEX fetch failed: ' + r.status + ' ' + url);
      const j = await r.json();
      const rows = j[dataKey] && j[dataKey].data;
      if (!rows || rows.length === 0) break;
      out.push.apply(out, rows);
      if (rows.length < 100) break;
      start += rows.length;
    }
    return out;
  }

  TCN.fetchOHLCV = async function (ticker, fromIso, tillIso) {
    const url = MOEX_BASE +
      '/history/engines/stock/markets/shares/securities/' + ticker + '.json' +
      '?from=' + fromIso + '&till=' + tillIso +
      '&iss.meta=off&iss.only=history' +
      '&history.columns=TRADEDATE,OPEN,HIGH,LOW,CLOSE,VOLUME,VALUE';
    const rows = await fetchPaged(url, 'history');
    const dates = []; const open = []; const high = []; const low = []; const close = []; const vol = [];
    for (const r of rows) {
      const d = r[0]; const c = r[4];
      if (!d || c === null) continue;
      dates.push(dateToMs(d));
      open.push(r[1] || 0);
      high.push(r[2] || 0);
      low.push(r[3] || 0);
      close.push(r[4]);
      vol.push(r[5] || 0);
    }
    return { dates: dates, open: open, high: high, low: low, close: close, volume: vol };
  };

  TCN.fetchIMOEX = async function (fromIso, tillIso) {
    const url = MOEX_BASE +
      '/history/engines/stock/markets/index/securities/IMOEX.json' +
      '?from=' + fromIso + '&till=' + tillIso +
      '&iss.meta=off&iss.only=history&history.columns=TRADEDATE,CLOSE';
    const rows = await fetchPaged(url, 'history');
    const dates = []; const values = [];
    for (const r of rows) {
      if (!r[0] || r[1] === null || r[1] === 0) continue;
      dates.push(dateToMs(r[0]));
      values.push(r[1]);
    }
    return { dates: dates, values: values };
  };

  TCN.fetchDividends = async function (secid) {
    const url = MOEX_BASE +
      '/securities/' + secid + '/dividends.json' +
      '?iss.meta=off&iss.only=dividends' +
      '&dividends.columns=secid,registryclosedate,value,currencyid';
    const r = await fetch(url);
    if (!r.ok) return { dates: [], values: [] };
    const j = await r.json();
    const rows = j.dividends && j.dividends.data;
    if (!rows) return { dates: [], values: [] };
    const dates = []; const values = [];
    for (const row of rows) {
      const cur = row[3];
      if (cur && cur !== 'RUB') continue;
      const d = row[1]; const v = row[2];
      if (!d || v === null || v === undefined) continue;
      dates.push(dateToMs(d));
      values.push(Number(v));
    }
    // sort ascending by date
    const order = dates.map((d, i) => i).sort((a, b) => dates[a] - dates[b]);
    return {
      dates: order.map(i => dates[i]),
      values: order.map(i => values[i]),
    };
  };

  // Build a {dates, close, ...} merge of IMOEX from MOEX.
  function isoDate(d) {
    const dt = new Date(d);
    const y = dt.getUTCFullYear();
    const m = String(dt.getUTCMonth() + 1).padStart(2, '0');
    const dd = String(dt.getUTCDate()).padStart(2, '0');
    return y + '-' + m + '-' + dd;
  }

  // ---- High-level helpers used by the UI -------------------------------
  function buildFactors(lastFeat) {
    // last_feat: 28-element row, see FEATURE_COLS for order.
    const logret_5 = lastFeat[3];
    const vol_rel = lastFeat[7];
    const rsi = lastFeat[9];
    const volatility = lastFeat[13];
    const div_yield = lastFeat[14];
    const usd_lr1 = lastFeat[17];
    const imoex_lr1 = lastFeat[22];

    const clip = v => Math.min(100, Math.max(0, v | 0));
    return [
      { key: 'volatility', label: 'Волатильность',         desc: 'уровень риска',
        impact: clip(Math.abs(volatility) * 2000), positive: volatility < 0.025 },
      { key: 'trend',      label: 'Тренд цены (5д)',        desc: 'направление движения',
        impact: clip(Math.abs(logret_5) * 1000),    positive: logret_5 > 0 },
      { key: 'volume',     label: 'Объём торгов',           desc: 'рыночная активность',
        impact: clip(Math.abs(vol_rel - 1.0) * 120), positive: vol_rel > 1.0 },
      { key: 'dividend',   label: 'Дивидендная доходность', desc: 'TTM',
        impact: clip(div_yield * 600),               positive: div_yield > 0.04 },
      { key: 'imoex',      label: 'Рынок (IMOEX, 1д)',      desc: 'фон рынка',
        impact: clip(Math.abs(imoex_lr1) * 5000),    positive: imoex_lr1 > 0 },
      { key: 'usd',        label: 'Курс USD/RUB (1д)',      desc: 'валютный фактор',
        impact: clip(Math.abs(usd_lr1) * 5000),      positive: usd_lr1 < 0 },
      { key: 'rsi',        label: 'RSI (14)',               desc: 'импульс',
        impact: clip(Math.abs(rsi - 50) * 2),        positive: rsi > 30 && rsi < 70 },
    ];
  }
  TCN.buildFactors = buildFactors;

  TCN.buildFeatures = buildFeatures;
  TCN._helpers = {
    rollingMean: rollingMean, rollingStd: rollingStd, rollingMax: rollingMax, rollingMin: rollingMin,
    ffill: ffill, logretClipped: logretClipped, rsi: rsi,
    mergeBackward: mergeBackward, ttmDivYield: ttmDivYield,
  };

  TCN.fetchAllForTicker = async function (ticker) {
    // Need ~280 trading days (≈ 13 months) before the current date so SMA200
    // and 200-day rolling features have valid values for the most recent rows.
    const now = new Date();
    const till = isoDate(now);
    const from = isoDate(now.getTime() - 730 * ONE_DAY_MS); // 2 calendar years
    const [price, imoex, divs] = await Promise.all([
      TCN.fetchOHLCV(ticker, from, till),
      TCN.fetchIMOEX(from, till),
      TCN.fetchDividends(ticker),
    ]);
    return { price: price, imoex: imoex, divs: divs };
  };

  window.TCN = TCN;
})();
