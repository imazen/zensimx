import init, * as wasm from "./pkg/zensim_workbench.js";
await init();
let atlas, lens;
self.onmessage = ({ data: { id, op, args } }) => {
  try {
    let value;
    if (op === "digest") value = wasm.blob_sha256(args[0]);
    else if (op === "pair") {
      const next = new wasm.AtlasSession("A", ...args);
      atlas?.free();
      atlas = next;
      value = JSON.parse(atlas.summary());
    } else if (op === "basic") value = JSON.parse(atlas.basic(...args));
    else if (op === "map") value = JSON.parse(atlas.detector(args[0]));
    else if (op === "bank") {
      const next = new wasm.LensSession("A", ...args);
      lens?.free();
      lens = next;
      value = true;
    } else if (op === "family") value = JSON.parse(lens.family(...args));
    else if (op === "diagnostics")
      value = JSON.parse(lens.diagnostics(...args));
    else if (op === "curve") value = JSON.parse(lens.curve(...args));
    else throw Error("Unknown operation");
    self.postMessage({ id, value });
  } catch (e) {
    self.postMessage({ id, error: String(e) });
  }
};
self.postMessage({
  ready: true,
  threads: wasm.thread_count(),
  metrics: JSON.parse(wasm.available_metrics()),
  bakeSha: wasm.bake_sha256("A"),
  metricSourceHash: wasm.metric_source_sha256(),
});
