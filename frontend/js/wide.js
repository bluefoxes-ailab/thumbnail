import { getImage, loadImage } from "./net.js";

// How many frames' pre-crop drag views stay resident. Each holds a decoded
// bitmap of the whole pre-crop frame — several megabytes of GPU/heap memory
// apiece, and twenty of them (one per grid slot) added up to hundreds. The
// user only ever drags one frame at a time, and a re-fetch on return is a
// single cached-on-the-backend request.
const WIDE_CACHE_SIZE = 4;

const _cache = new Map();   // frame_id -> {url, meta fields, _img}

// How many times each frame's view has been thrown away, plus a counter for
// the whole cache being dropped. Together they stamp a fetch with the state it
// was started under.
//
// Dropping what's in the map is not enough on its own: a /frame-wide request
// is in flight for several seconds on a frame whose restored base still has to
// be built, and the fire-and-forget prefetch at the end of every enhance pass
// means one is usually running right when the user clicks Variation. That
// request was answered from the photo the slot pointed at when it was SENT, so
// letting it land after the swap put the previous photo's pre-crop view back
// in the cache under the new photo's frame_id — and the next drag drew its
// ghost from exactly that. The frame the user had just replaced reappeared
// under their cursor while reframing.
let _generation = 0;
const _epochs = new Map();

const stamp = (frameId) => `${_generation}:${_epochs.get(frameId) || 0}`;

function touch(frameId, entry) {
    _cache.delete(frameId);
    _cache.set(frameId, entry);
    while (_cache.size > WIDE_CACHE_SIZE) {
        const oldestId = _cache.keys().next().value;
        release(oldestId);
    }
    return entry;
}

function release(frameId) {
    const entry = _cache.get(frameId);
    if (!entry) return;
    if (entry.url) URL.revokeObjectURL(entry.url);
    entry._img = null;
    _cache.delete(frameId);
}

export function peek(frameId) {
    return _cache.get(frameId) || null;
}

/**
 * Which photo/preset this frame's view currently describes. A gesture that
 * armed itself from a view can compare this against the stamp it started with
 * to find out whether the slot moved on underneath it — a drag started while a
 * Variation request is still out is panning the photo about to be replaced,
 * and the window it drops belongs to a photo that no longer exists.
 */
export const viewStamp = (frameId) => stamp(frameId);

export function invalidate(frameId) {
    _epochs.set(frameId, (_epochs.get(frameId) || 0) + 1);
    release(frameId);
}

export function clearAll() {
    _generation++;   // frame_ids are reused per video — see the note on _epochs
    for (const id of [..._cache.keys()]) release(id);
    _epochs.clear();
}

export async function fetchWide(frameId) {
    const hit = _cache.get(frameId);
    if (hit) return touch(frameId, hit);
    const started = stamp(frameId);
    try {
        const res = await getImage(`/frame-wide/${frameId}`);
        if (!res) return null;
        if (stamp(frameId) !== started) {
            // This slot moved onto a different photo (or a different edit
            // preset, or a whole different video) while the request was out —
            // what came back describes the one it left behind.
            URL.revokeObjectURL(res.url);
            return null;
        }
        return touch(frameId, { url: res.url, ...res.meta, _img: null });
    } catch (_) {
        return null;
    }
}

/**
 * Decodes the wide view's bytes into an <img> usable as a drawImage source —
 * cached on the record itself so a re-drag doesn't re-decode.
 */
export async function loadWideImage(wide) {
    if (wide._img) return wide._img;
    try {
        wide._img = await loadImage(wide.url);
        return wide._img;
    } catch (_) {
        return null;
    }
}

/**
 * Warms the cache right after a frame's restored pixels are ready, instead of
 * waiting for the user's first drag to trigger the fetch+decode live — that
 * lazy path is what made the first reframe drag on a frame (or the first one
 * after a preset switch) feel like a fresh loading beat. Fire-and-forget.
 */
export async function prefetchWide(frameId) {
    const wide = await fetchWide(frameId);
    if (wide) await loadWideImage(wide);
}
