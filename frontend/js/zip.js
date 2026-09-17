/**
 * zip.js — Packing a handful of finished images into one file to download.
 *
 * A capture run hands back forty stills, and forty separate browser downloads
 * is not a delivery, it is forty dialogs and a folder the user has to sort
 * out afterwards. One archive is what "download all" has to mean.
 *
 * Written here rather than vendored, for two reasons. The first is that the
 * whole of what is needed is below: a ZIP holding already-compressed files is
 * a header, the bytes, and an index — the format's "stored" method, which is
 * no compression at all. PNG and JPEG are compressed already, so deflating
 * them a second time costs CPU and saves nothing measurable. The second is
 * that this app deliberately ships without a CDN (see the note at the top of
 * index.html): another library would be another file to vendor, update and
 * explain, for eighty lines of byte-writing.
 *
 * What this deliberately does NOT support, because nothing here needs it:
 * directories, compression, encryption, Zip64 (so: under 4GB total, under
 * 65535 entries), or names outside ASCII. The last one is not a limitation
 * anybody meets — the names are "frame 01.png".
 */

// Every offset and size in the format is little-endian, and the two headers
// below are the only structures written. Field order is the specification's.
const LOCAL_HEADER_SIG = 0x04034b50;
const CENTRAL_HEADER_SIG = 0x02014b50;
const END_OF_CENTRAL_SIG = 0x06054b50;
// "Stored" — the entry's bytes are the file's bytes. See the header.
const METHOD_STORED = 0;
// The version-needed field. 2.0 is what a stored entry requires, and is what
// every reader in existence handles.
const VERSION_NEEDED = 20;

/**
 * CRC-32, which every entry carries and which readers check before unpacking.
 *
 * The table is built once on first use rather than written out as a literal:
 * it is 256 numbers derived from one polynomial, and the derivation is
 * shorter, verifiable by eye, and cannot contain a typo that only shows up on
 * one particular file.
 */
let _crcTable = null;

function crcTable() {
    if (_crcTable) return _crcTable;
    _crcTable = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
        let c = n;
        for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
        _crcTable[n] = c >>> 0;
    }
    return _crcTable;
}

function crc32(bytes) {
    const table = crcTable();
    let c = 0xFFFFFFFF;
    for (let i = 0; i < bytes.length; i++) c = table[(c ^ bytes[i]) & 0xFF] ^ (c >>> 8);
    return (c ^ 0xFFFFFFFF) >>> 0;
}

/**
 * MS-DOS date and time, which is what a ZIP entry's timestamp is: two 16-bit
 * words, seconds stored in two-second steps, years counted from 1980.
 *
 * Coarse by the format's own definition, not by shortcut here — there is
 * nowhere in a base ZIP entry to put anything finer.
 */
function dosDateTime(date) {
    const time = (date.getHours() << 11) | (date.getMinutes() << 5) | (date.getSeconds() >> 1);
    const day = ((date.getFullYear() - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate();
    return { time: time & 0xFFFF, date: day & 0xFFFF };
}

/** ASCII bytes of `name`. See the header for why the alphabet is not a problem here. */
function nameBytes(name) {
    const out = new Uint8Array(name.length);
    for (let i = 0; i < name.length; i++) out[i] = name.charCodeAt(i) & 0x7F;
    return out;
}

/** A little-endian writer over a fixed-size buffer — the whole of what the format needs. */
function writer(size) {
    const bytes = new Uint8Array(size);
    const view = new DataView(bytes.buffer);
    let at = 0;
    return {
        bytes,
        u16(v) { view.setUint16(at, v, true); at += 2; },
        u32(v) { view.setUint32(at, v >>> 0, true); at += 4; },
        raw(src) { bytes.set(src, at); at += src.length; },
    };
}

/**
 * One archive holding `files`, as a Blob ready to hand to a download.
 *
 * Each file is `{ name, blob }`, and they are written in the order given —
 * which for a capture run is the order the frames occur in the video, so an
 * unpacked folder reads the same way the grid did.
 */
export async function zipBlob(files) {
    const stamp = dosDateTime(new Date());
    const entries = [];
    let offset = 0;
    const parts = [];

    for (const file of files) {
        const data = new Uint8Array(await file.blob.arrayBuffer());
        const name = nameBytes(file.name);
        const crc = crc32(data);

        const head = writer(30 + name.length);
        head.u32(LOCAL_HEADER_SIG);
        head.u16(VERSION_NEEDED);
        head.u16(0);                 // no flags: not encrypted, sizes known up front
        head.u16(METHOD_STORED);
        head.u16(stamp.time);
        head.u16(stamp.date);
        head.u32(crc);
        head.u32(data.length);       // compressed size — the same, being stored
        head.u32(data.length);
        head.u16(name.length);
        head.u16(0);                 // no extra field
        head.raw(name);

        parts.push(head.bytes, data);
        entries.push({ name, crc, size: data.length, offset });
        offset += head.bytes.length + data.length;
    }

    // The index. A reader finds this first (from the end of the file) and
    // never has to walk the entries, which is why every entry's offset is
    // repeated here.
    const directoryStart = offset;
    let directorySize = 0;
    for (const entry of entries) {
        const head = writer(46 + entry.name.length);
        head.u32(CENTRAL_HEADER_SIG);
        head.u16(VERSION_NEEDED);    // version made by
        head.u16(VERSION_NEEDED);
        head.u16(0);
        head.u16(METHOD_STORED);
        head.u16(stamp.time);
        head.u16(stamp.date);
        head.u32(entry.crc);
        head.u32(entry.size);
        head.u32(entry.size);
        head.u16(entry.name.length);
        head.u16(0);                 // extra field
        head.u16(0);                 // comment
        head.u16(0);                 // disk number — one file, one disk
        head.u16(0);                 // internal attributes
        head.u32(0);                 // external attributes
        head.u32(entry.offset);
        head.raw(entry.name);

        parts.push(head.bytes);
        directorySize += head.bytes.length;
    }

    const end = writer(22);
    end.u32(END_OF_CENTRAL_SIG);
    end.u16(0);                      // this disk
    end.u16(0);                      // disk the directory starts on
    end.u16(entries.length);
    end.u16(entries.length);
    end.u32(directorySize);
    end.u32(directoryStart);
    end.u16(0);                      // archive comment
    parts.push(end.bytes);

    return new Blob(parts, { type: "application/zip" });
}
