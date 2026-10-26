// Original author JS actor remains an independent read-only numerical reference.
import fs from 'fs';
import path from 'path';
import {pathToFileURL} from 'url';
const [assets,input] = process.argv.slice(2);
const {HybridPolicy} = await import(pathToFileURL(path.resolve(assets,'src/hybrid.js')).href);
const manifest = JSON.parse(fs.readFileSync(path.join(assets,'assets/hybrid.json')));
const raw = fs.readFileSync(path.join(assets,'assets/hybrid.bin'));
const weights = new Float32Array(raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength));
const policy = new HybridPolicy(manifest, weights);
const fixture = JSON.parse(fs.readFileSync(input));
console.log(JSON.stringify(fixture.obs.map((obs,i)=>Array.from(policy.logits(
  new Float32Array(obs),fixture.mask[i],new Float32Array(fixture.dodge[i]))))));
