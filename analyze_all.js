/**
 * 摄影眼 analysis — all images, both P1 modes, with debug visualization.
 *
 * Run: node assets/eye/analyze_all.js
 */
const fs = require('fs');
const pako = require('pako');

// ─── PNG I/O ────────────────────────────────────────────────────
function u32(b, o) { return ((b[o]<<24)|(b[o+1]<<16)|(b[o+2]<<8)|b[o+3])>>>0; }

function readPNG(path) {
  const bytes = new Uint8Array(fs.readFileSync(path));
  let off = 8, w = 0, h = 0, ct = 2, idats = [];
  while (off < bytes.length) {
    const len = u32(bytes, off);
    const type = String.fromCharCode(bytes[off+4],bytes[off+5],bytes[off+6],bytes[off+7]);
    const data = bytes.subarray(off+8, off+8+len);
    if (type === 'IHDR') { w = u32(data,0); h = u32(data,4); ct = data[9]; }
    else if (type === 'IDAT') idats.push(data);
    else if (type === 'IEND') break;
    off += 12 + len;
  }
  const total = idats.reduce((s,c)=>s+c.length,0), comb = new Uint8Array(total);
  let pos = 0;
  for (const c of idats) { comb.set(c, pos); pos += c.length; }
  const raw = pako.inflate(comb), bpp = ct === 6 ? 4 : ct === 2 ? 3 : 1;
  const pixels = new Uint8Array(w * h * 3);
  let si = 0, di = 0;
  for (let y = 0; y < h; y++) { si += 1;
    for (let x = 0; x < w; x++) {
      pixels[di++] = raw[si++]; pixels[di++] = raw[si++]; pixels[di++] = raw[si++];
      if (bpp === 4) si++;
    }
  }
  return { pixels, w, h, ct, bpp, raw };
}

function makeThumb(fullPx, fw, fh) {
  const tw = Math.min(64, fw), th = Math.round(fh * (tw / fw));
  const p = new Uint8Array(tw * th * 3);
  for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) {
    const si = (Math.floor(y*fh/th)*fw + Math.floor(x*fw/tw)) * 3;
    const di = (y*tw + x) * 3;
    p[di]=fullPx[si]; p[di+1]=fullPx[si+1]; p[di+2]=fullPx[si+2];
  }
  return { pixels: p, width: tw, height: th };
}

// ─── Heatmap peak detection ─────────────────────────────────────
function findPeaks(pixels, p1Mode) {
  const hmW = 32, hmH = Math.round(pixels.height * (hmW / pixels.width));
  const heat = new Float32Array(hmW * hmH);
  const sx = pixels.width / hmW, sy = pixels.height / hmH;

  for (let y = 0; y < hmH; y++) for (let x = 0; x < hmW; x++) {
    const px = Math.floor((x+0.5)*sx), py = Math.floor((y+0.5)*sy);
    const idx = (py*pixels.width + px) * 3;
    const r = pixels.pixels[idx]/255, g = pixels.pixels[idx+1]/255, b = pixels.pixels[idx+2]/255;
    const lum = 0.299*r + 0.587*g + 0.114*b;
    if (p1Mode === 'warmth') {
      const warmth = (pixels.pixels[idx] - pixels.pixels[idx+2]) / 255;
      const ws = Math.max(0, Math.min(1, (warmth+0.3)/0.6));
      heat[y*hmW+x] = lum*0.6 + ws*0.4;
    } else {
      heat[y*hmW+x] = lum;
    }
  }

  // Box blur 5x5
  const blurred = new Float32Array(hmW * hmH);
  for (let y = 0; y < hmH; y++) for (let x = 0; x < hmW; x++) {
    let s=0, c=0;
    for (let dy=-2; dy<=2; dy++) for (let dx=-2; dx<=2; dx++) {
      const ny=y+dy, nx=x+dx;
      if (ny>=0 && ny<hmH && nx>=0 && nx<hmW) { s+=heat[ny*hmW+nx]; c++; }
    }
    blurred[y*hmW+x] = s/c;
  }

  // Local maxima
  const thr = p1Mode === 'warmth' ? 0.15 : 0.08;
  const peaks = [];
  for (let y = 1; y < hmH-1; y++) for (let x = 1; x < hmW-1; x++) {
    const v = blurred[y*hmW+x];
    if (v < thr || isNaN(v)) continue;
    if (v >= blurred[(y-1)*hmW+x] && v >= blurred[(y+1)*hmW+x] &&
        v >= blurred[y*hmW+x-1] && v >= blurred[y*hmW+x+1] &&
        v >= blurred[(y-1)*hmW+x-1] && v >= blurred[(y-1)*hmW+x+1] &&
        v >= blurred[(y+1)*hmW+x-1] && v >= blurred[(y+1)*hmW+x+1]) {
      peaks.push({ x, y, v });
    }
  }
  peaks.sort((a,b) => b.v - a.v);

  // Keep diverse top 3
  const kept = [];
  for (const p of peaks) {
    let close = false;
    for (const k of kept) {
      if (Math.sqrt(((p.x-k.x)/hmW)**2 + ((p.y-k.y)/hmH)**2) < 0.2) { close=true; break; }
    }
    if (!close) { kept.push(p); if (kept.length >= 3) break; }
  }

  // Convert to composition boxes (~40% x 50% of frame)
  const regions = [];
  for (let i = 0; i < kept.length; i++) {
    const p = kept[i], cx = (p.x+0.5)/hmW, cy = (p.y+0.5)/hmH;
    let l = cx - 0.2, t = cy - 0.25;
    l = Math.max(0, Math.min(1-0.4, l));
    t = Math.max(0, Math.min(1-0.5, t));
    regions.push({ id: `peak_${i}`, left:l, top:t, right:l+0.4, bottom:t+0.5,
      source:'peak', peakValue:p.v, peakX:cx, peakY:cy });
  }

  return { regions, heatmap: blurred, heatW: hmW, heatH: hmH, allPeaks: peaks };
}

// ─── Region metrics ─────────────────────────────────────────────
function computeMetrics(region, pix, tw, th) {
  const pl = Math.max(0, Math.floor(region.left*tw));
  const pt = Math.max(0, Math.floor(region.top*th));
  const pr = Math.min(tw-1, Math.ceil(region.right*tw));
  const pb = Math.min(th-1, Math.ceil(region.bottom*th));
  if (pl >= pr || pt >= pb) return null;

  const lums = []; const h16 = new Float32Array(16);
  let satS = 0, satC = 0, rS = 0, bS = 0;
  for (let py = pt; py <= pb; py++) for (let px = pl; px <= pr; px++) {
    const idx = (py*tw + px) * 3;
    const r = pix[idx]/255, g = pix[idx+1]/255, b = pix[idx+2]/255;
    rS += pix[idx]; bS += pix[idx+2];
    const lum = 0.299*r + 0.587*g + 0.114*b; lums.push(lum);
    const maxC = Math.max(r,g,b), minC = Math.min(r,g,b), delta = maxC-minC;
    satS += maxC>0 ? delta/maxC : 0; satC++;
    if (delta > 0.01) {
      let hue=0;
      if (maxC===r) hue = 60*(((g-b)/delta)%6);
      else if (maxC===g) hue = 60*(((b-r)/delta)+2);
      else hue = 60*(((r-g)/delta)+4);
      if (hue<0) hue+=360;
      h16[Math.min(15, Math.floor(hue/22.5))]++;
    }
  }
  const n = lums.length, mean = n>0 ? lums.reduce((a,b)=>a+b,0)/n : 0;
  const variance = n>0 ? lums.reduce((sv,v)=>sv+(v-mean)**2,0)/n : 0;
  const lumStd = Math.sqrt(variance);
  let bimodal = false;
  if (n > 4) { const s = [...lums].sort((a,b)=>a-b), m = s[Math.floor(n/2)], lo = s.filter(v=>v<=m), hi = s.filter(v=>v>m);
    if (lo.length>0 && hi.length>0) bimodal = (hi.reduce((a,b)=>a+b,0)/hi.length - lo.reduce((a,b)=>a+b,0)/lo.length) > 0.25; }
  const tH = h16.reduce((s,v)=>s+v,0);
  let hueE = 0; if (tH>0) for (let i=0;i<16;i++) if (h16[i]>0) { const p=h16[i]/tH; hueE -= p*Math.log2(p); }
  const mSat = satC>0 ? satS/satC : 0, nPx = (pr-pl+1)*(pb-pt+1);
  return { lumMean: mean, luminanceStd: lumStd, luminanceBimodal: bimodal,
    warmth: nPx>0 ? (rS-bS)/(nPx*255) : 0, hueEntropy: hueE, meanSaturation: mSat };
}

// ─── Score regions ──────────────────────────────────────────────
function scoreRegions(regions, thumb, p1Mode) {
  return regions.map(r => {
    const m = computeMetrics(r, thumb.pixels, thumb.width, thumb.height);
    if (!m) return null;
    let p1Score, p1Label;
    if (p1Mode === 'warmth') {
      const ws = Math.max(0, Math.min(1, (m.warmth+0.3)/0.6));
      p1Score = m.lumMean*0.6 + ws*0.4;
      p1Label = m.warmth > 0.08 ? '阳光区域' : m.lumMean > 0.15 ? '明亮区域' : '光影柔和';
    } else {
      p1Score = m.lumMean;
      p1Label = m.lumMean > 0.25 ? '明亮区域' : m.lumMean > 0.15 ? '柔和光线' : '光影柔和';
    }
    return { region: r, score: p1Score, label: p1Label, metrics: m };
  }).filter(Boolean).sort((a,b) => b.score - a.score);
}

// ─── Debug PNG drawing ──────────────────────────────────────────
function drawDebug(img, peakResult, scored, p1Mode, outPath) {
  const { pixels: fullPx, w: fw, h: fh, bpp, ct, raw } = img;
  const rawCopy = new Uint8Array(raw.length);
  rawCopy.set(raw);
  const stride = fw * bpp + 1;

  function setPx(x, y, r, g, b) {
    if (x < 0 || x >= fw || y < 0 || y >= fh) return;
    const ofs = y*stride + 1 + x*bpp;
    rawCopy[ofs]=r; rawCopy[ofs+1]=g; rawCopy[ofs+2]=b;
    if (bpp===4) rawCopy[ofs+3]=255;
  }

  function drawBox(region, r, g, b, thick) {
    thick = thick||4;
    const x1 = Math.round(region.left*fw), x2 = Math.round(region.right*fw);
    const y1 = Math.round(region.top*fh), y2 = Math.round(region.bottom*fh);
    for (let y = y1; y <= y2; y++) for (let x = x1; x <= x2; x++)
      if (y<y1+thick || y>y2-thick || x<x1+thick || x>x2-thick) setPx(x,y,r,g,b);
  }

  function drawLabel(x, y, text, r, g, b) {
    const w = text.length*7+8, h=16;
    for (let dy=0; dy<h; dy++) for (let dx=0; dx<w; dx++) setPx(x+dx, y+dy, 0, 0, 0);
    for (let dy=2; dy<h-2; dy++) for (let dx=4; dx<w-4; dx+=6) setPx(x+dx, y+dy, r, g, b);
  }

  // ── Heatmap overlay (top-left, 3x scale) ──
  const { heatmap, heatW, heatH, allPeaks } = peakResult;
  const hScale = 3, hmX = 10, hmY = 10;
  for (let y = 0; y < heatH; y++) for (let x = 0; x < heatW; x++) {
    const v = heatmap[y*heatW+x];
    const intensity = Math.min(255, Math.round(v*255*3));
    for (let dy=0; dy<hScale; dy++) for (let dx=0; dx<hScale; dx++)
      setPx(hmX+x*hScale+dx, hmY+y*hScale+dy, intensity, Math.round(intensity*0.3), Math.round(intensity*0.1));
  }
  drawLabel(hmX, hmY+heatH*hScale+2, `HEATMAP (${p1Mode})`, 255, 200, 100);

  // ── All peaks (yellow dots) ──
  for (const p of allPeaks) {
    const cx = Math.round((p.x+0.5)/heatW*fw), cy = Math.round((p.y+0.5)/heatH*fh);
    for (let dy=-5; dy<=5; dy++) for (let dx=-5; dx<=5; dx++)
      if (dx*dx+dy*dy<=25) setPx(cx+dx, cy+dy, 255, 255, 0);
  }

  // ── Kept peaks (green dots, larger) ──
  for (const r of peakResult.regions) {
    const cx = Math.round(r.peakX*fw), cy = Math.round(r.peakY*fh);
    for (let dy=-8; dy<=8; dy++) for (let dx=-8; dx<=8; dx++)
      if (dx*dx+dy*dy<=64) setPx(cx+dx, cy+dy, 0, 255, 0);
  }

  // ── Top 2 boxes ──
  if (scored.length >= 1) {
    const s = scored[0];
    drawBox(s.region, 0, 255, 0, 5);
    const x1 = Math.round(s.region.left*fw);
    const y2 = Math.round(s.region.bottom*fh);
    drawLabel(x1, y2+4, `#1 ${s.label} score=${s.score.toFixed(3)} lum=${s.metrics.lumMean.toFixed(3)} w=${s.metrics.warmth.toFixed(3)}`, 0, 255, 0);
  }
  if (scored.length >= 2) {
    const s = scored[1];
    drawBox(s.region, 100, 150, 255, 3);
    const x1 = Math.round(s.region.left*fw);
    const y2 = Math.round(s.region.bottom*fh);
    drawLabel(x1, y2+4, `#2 ${s.label} score=${s.score.toFixed(3)}`, 100, 150, 255);
  }

  // ── Write PNG ──
  const rawData = new Uint8Array(stride * fh);
  for (let y = 0; y < fh; y++) {
    rawData[y*stride] = 0;
    for (let x = 0; x < fw; x++) {
      const si = y*stride + 1 + x*bpp;
      rawData[y*stride+1+x*bpp] = rawCopy[si];
      if (bpp>=2) rawData[y*stride+1+x*bpp+1] = rawCopy[si+1];
      if (bpp>=3) rawData[y*stride+1+x*bpp+2] = rawCopy[si+2];
    }
  }
  const compressed = pako.deflate(rawData);
  function chunk(t, d) {
    const l = Buffer.alloc(4); l.writeUInt32BE(d.length);
    const tb = Buffer.from(t, 'ascii'), data = Buffer.concat([tb, d]);
    let crc = 0xFFFFFFFF;
    for (let i = 0; i < data.length; i++) { crc ^= data[i]; for (let j=0; j<8; j++) crc = (crc>>>1) ^ (crc&1 ? 0xEDB88320 : 0); }
    const cb = Buffer.alloc(4); cb.writeUInt32BE((crc^0xFFFFFFFF)>>>0);
    return Buffer.concat([l, tb, d, cb]);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(fw, 0); ihdr.writeUInt32BE(fh, 4);
  ihdr[8]=8; ihdr[9]=ct;
  const out = Buffer.concat([
    Buffer.from([137,80,78,71,13,10,26,10]),
    chunk('IHDR', ihdr), chunk('IDAT', Buffer.from(compressed)), chunk('IEND', Buffer.alloc(0))
  ]);
  fs.writeFileSync(outPath, out);
}

// ─── Main ───────────────────────────────────────────────────────
const images = ['library.png', 'bridge.png', 'building.png', 'contrast.png'];
const modes = ['brightness', 'warmth'];

console.log('\n╔══════════════════════════════════════════════════════════════╗');
console.log('║       摄影眼 构图分析 - Heatmap 峰值法                   ║');
console.log('╚══════════════════════════════════════════════════════════════╝\n');

for (const imgName of images) {
  const imgPath = 'assets/eye/' + imgName;
  if (!fs.existsSync(imgPath)) { console.log(`SKIP: ${imgPath}`); continue; }

  const img = readPNG(imgPath);
  const thumb = makeThumb(img.pixels, img.w, img.h);
  console.log(`📷 ${imgName} (${img.w}×${img.h})  thumb: ${thumb.width}×${thumb.height}`);

  for (const mode of modes) {
    const peakResult = findPeaks(thumb, mode);
    const scored = scoreRegions(peakResult.regions, thumb, mode);

    console.log(`  ── ${mode === 'brightness' ? '亮度优先' : '阳光感优先'}`);

    if (scored.length === 0) {
      console.log('    无推荐\n');
      continue;
    }

    for (let i = 0; i < Math.min(scored.length, 3); i++) {
      const s = scored[i];
      const r = s.region;
      const cx = ((r.left+r.right)/2*100).toFixed(1);
      const cy = ((r.top+r.bottom)/2*100).toFixed(1);
      console.log(`    #${i+1} [${r.id}] score=${s.score.toFixed(3)} ${s.label}`);
      console.log(`       中心(${cx}%,${cy}%)  lum=${s.metrics.lumMean.toFixed(3)}  warmth=${s.metrics.warmth.toFixed(3)}`);
      console.log(`       框: L=${r.left.toFixed(3)} T=${r.top.toFixed(3)} R=${r.right.toFixed(3)} B=${r.bottom.toFixed(3)}`);
    }

    // Draw and save debug image
    const outName = imgName.replace('.png', `_${mode}_analysis.png`);
    drawDebug(img, peakResult, scored, mode, 'assets/eye/' + outName);
    console.log(`   ✅ 已保存: ${outName}\n`);
  }
}

// ─── Summary table ──────────────────────────────────────────────
console.log('══════════════════════════════════════════════════════════════');
console.log('  总结');
console.log('══════════════════════════════════════════════════════════════\n');

for (const imgName of images) {
  const imgPath = 'assets/eye/' + imgName;
  if (!fs.existsSync(imgPath)) continue;
  const img = readPNG(imgPath);
  const thumb = makeThumb(img.pixels, img.w, img.h);
  console.log(`  ${imgName}:`);

  for (const mode of modes) {
    const peakResult = findPeaks(thumb, mode);
    const scored = scoreRegions(peakResult.regions, thumb, mode);
    if (scored.length === 0) { console.log(`    ${mode}: 无推荐`); continue; }
    const s = scored[0];
    const cx = ((s.region.left+s.region.right)/2*100).toFixed(1);
    const cy = ((s.region.top+s.region.bottom)/2*100).toFixed(1);
    console.log(`    ${mode==='brightness'?'亮度':'温暖'}: "${s.label}" score=${s.score.toFixed(3)} 中心(${cx},${cy}) lum=${s.metrics.lumMean.toFixed(3)} w=${s.metrics.warmth.toFixed(3).padStart(6)}`);
  }
  console.log('');
}
