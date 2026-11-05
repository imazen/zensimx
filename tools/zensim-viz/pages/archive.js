// The native IMIJ-set writer puts an application catalog in its first blob.
// This reader supports that immutable, single-generation bundle profile only.
// Entry offsets come from the verified native archive, never from filenames.
export class VizArchive {
  static async open(url, digest) {
    const self = new VizArchive();
    self.url = url;
    self.digest = digest;
    const bytes = await self.range(0, 65536);
    const catalog = JSON.parse(
      new TextDecoder("utf-8", { fatal: true }).decode(bytes).trim(),
    );
    if (
      catalog.schema !== "zensim-imijset-v1" ||
      catalog.bootstrap_bytes !== 65536 ||
      catalog.generation !== 1 ||
      !Array.isArray(catalog.entries)
    )
      throw Error("Unsupported IMIJ-set catalog");
    const tail = await self.range(catalog.bytes - 80, 80);
    if (new TextDecoder().decode(tail.slice(0, 8)) !== "IMIJSET\x01")
      throw Error("Missing IMIJ-set footer");
    const footer = new DataView(tail.buffer, tail.byteOffset, tail.byteLength);
    if (
      Number(footer.getBigUint64(8, true)) !== catalog.bytes ||
      Number(footer.getBigUint64(64, true)) !== 1 ||
      Number(footer.getBigUint64(48, true)) !== catalog.entries.length + 1
    )
      throw Error("IMIJ-set generation/length/count mismatch");
    self.catalog = catalog;
    self.cache = new Map();
    let end = 65536;
    for (const e of catalog.entries) {
      if (
        typeof e.name !== "string" ||
        !Number.isSafeInteger(e.offset) ||
        !Number.isSafeInteger(e.length) ||
        e.offset < end ||
        e.length <= 0 ||
        e.offset + e.length > catalog.bytes - 80 ||
        !/^[0-9a-f]{64}$/.test(e.sha256)
      )
        throw Error("Invalid IMIJ-set catalog entry");
      end = e.offset + e.length;
    }
    return self;
  }
  async range(offset, length) {
    const end = offset + length - 1;
    const r = await fetch(this.url, {
      headers: { Range: `bytes=${offset}-${end}` },
    });
    if (r.status !== 206)
      throw Error(`Archive requires HTTP 206; got ${r.status}`);
    const b = new Uint8Array(await r.arrayBuffer());
    if (b.length !== length) throw Error("Truncated archive range");
    return b;
  }
  async bytes(name) {
    if (this.cache.has(name)) return this.cache.get(name);
    const e = this.catalog.entries.find((e) => e.name === name);
    if (!e) throw Error(`No archive entry ${name}`);
    const bytes = await this.range(e.offset, e.length);
    if ((await this.digest(bytes)) !== e.sha256)
      throw Error(`Archive entry checksum differs: ${name}`);
    this.cache.set(name, bytes);
    return bytes;
  }
  async json(name) {
    let bytes = await this.bytes(name);
    if (name.endsWith(".gz"))
      bytes = new Uint8Array(
        await new Response(
          new Blob([bytes])
            .stream()
            .pipeThrough(new DecompressionStream("gzip")),
        ).arrayBuffer(),
      );
    return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  }
}
