import { DEFAULT_PRESET } from "./config.js";
import { branding, activeChannelId, styleFor, captureFor, decorFor } from "./channels.js";

// [{frame_id, plainUrl, enhancedUrl, lines, hl, titleBox, titleActive, glow,
//   flipImage, flipText, gradient, gradientTouched, logos, activeLogo,
//   editPreset, picked, edited, version}]
// The title is per-frame — each thumbnail carries its own lines, its own
// highlights and, where its channel allows it, its own placement and halo —
// so the user works through frames one at a time and every edit stays with
// the frame it was made on.
export const store = {
    frames: [],
    selected: 0,
};

export const frames = () => store.frames;
export const current = () => store.frames[store.selected];
export const selectedIndex = () => store.selected;
export const setSelectedIndex = (i) => { store.selected = i; };

/**
 * The lines of frame `i`'s title, as the renderer will set them.
 *
 * Two shapes, because two kinds of channel write titles two different ways
 * (see `freeform` in channels.js), and the difference is entirely in what an
 * EMPTY line means.
 *
 * With three fixed slots it means "this slot is unused": the block is
 * whichever of them were filled in, and leaving the middle one blank gives a
 * two-line title rather than a gap. With one block of text the breaks are the
 * user's own, so a blank line between two others is a blank line they typed
 * and it is kept — the renderer sets it as a spacer row of its own height.
 * Only the blanks at the two ENDS are dropped, which is what makes a trailing
 * newline harmless: nobody types one meaning to push the block up the canvas.
 *
 * The channel is the frame's, not the page's — this is asked for frames that
 * are not the one on screen (the grid, a batch download), and the shape of a
 * stored title belongs to the brand it was written under.
 */
export function linesFor(i) {
    const f = store.frames[i];
    const raw = (f && f.lines) ? f.lines : [];
    const layout = styleFor(channelFor(i)).layout;
    if (!layout.freeform) {
        // Only the slots this channel HAS. A frame carries one set of lines
        // whatever brand it is being looked at under (which is what lets the
        // user try another one without losing the title), so a four-line title
        // moved onto a three-line channel would otherwise keep drawing a
        // fourth line the panel no longer offers any way to edit or clear.
        return raw.slice(0, layout.lines).map(l => l.trim()).filter(l => l);
    }
    // Blank lines are dropped from the two ENDS of a freeform block — an
    // untouched frame's stored ["", "", ""] is not a three-line title, and a
    // stray return at the end of the box is not a fourth line. Blank lines
    // BETWEEN two others are the user's own and stay (see the spacer rows in
    // text.js).
    //
    // The lines themselves are NOT trimmed, and that is a decision rather
    // than an omission. On a channel that draws a box behind each line, the
    // spaces at the end of one are how the user asks for room inside that box
    // — for an emoji, for a gap the sentence does not fill on its own (see
    // `panel.scope` in channels.js). Trimming them here would make the
    // request unanswerable before anything could read it. Everywhere else
    // they cost nothing: a space has no ink, and every measurement in the
    // renderer but that one is taken from the ink.
    const lines = [...raw];
    while (lines.length && !lines[0].trim()) lines.shift();
    while (lines.length && !lines[lines.length - 1].trim()) lines.pop();
    return lines;
}

/**
 * The channel frame `i` is branded with — its font, colours and layout.
 *
 * Per frame, not per page: see the note at the top of channels.js. A frame
 * created before any channel was picked carries the empty selection, which
 * draws no title at all.
 */
export function channelFor(i) {
    const f = store.frames[i];
    return (f && f.channel) || "";
}

/**
 * What is highlighted in frame `i`'s title.
 *
 * The ELEMENTS are whatever the frame's channel highlights (see `scope` in
 * channels.js): line indices into linesFor(i) for a channel that highlights
 * whole lines, and "line:word" keys for one that highlights words. One field
 * either way, because it is one question — what in this title is picked out —
 * and a frame only ever has one channel to answer it for.
 *
 * The two shapes cannot be mistaken for each other, which is what makes
 * switching a frame between two such channels harmless: a word key is a
 * string and matches no line index, a line index is a number and matches no
 * word, so each channel simply sees nothing highlighted until something is.
 * The default is line 0, which is that same nothing for a word channel.
 */
export function hlFor(i) {
    const f = store.frames[i];
    return f && Array.isArray(f.hl) ? f.hl : [0];
}

/**
 * Where frame `i`'s title has been dragged and how far it has been resized,
 * or null while it is still where its channel's guidelines put it.
 *
 * An OFFSET from the computed layout rather than an absolute position: the
 * guidelines still decide where the block starts, how wide it is set and how
 * its lines stack, and this says how far the user has since moved it from
 * there. Which means a title that has been nudged 30px is still laid out by
 * its brand — and that clearing this puts it back exactly where the channel
 * would have put it, with nothing to recompute.
 */
export function titleBoxFor(i) {
    const f = store.frames[i];
    return (f && f.titleBox) || null;
}

/** Whether the frame's title carries the on-canvas handles (see activeLogoFor). */
export function titleActiveFor(i) {
    const f = store.frames[i];
    return !!(f && f.titleActive);
}

/**
 * The figures frame `i` shows, bottom of the stack first — array order IS
 * stacking order, exactly as it is for the image overlays.
 *
 * Each entry is one cut-out person AND where they go: `src`/`natW`/`natH` are
 * the backend's answer (see /cutout-frame), `x`/`y`/`scale` are the placement,
 * and `flip` mirrors that figure alone.
 *
 * Usually one. Five of this channel's twenty slots carry two and five carry
 * three (see `copiesBySlot` in js/channels.js), and each of those is a
 * DIFFERENT moment of the same shot rather than a second print of the same
 * photograph — the subject in three poses, not one pose three times.
 */
export function layersFor(i) {
    const f = store.frames[i];
    return (f && Array.isArray(f.layers)) ? f.layers : [];
}

/**
 * The layer the on-canvas reorder arrows belong to, and the one the zoom and
 * the flip act on — or null.
 *
 * The same question activeLogoFor answers about the overlays, and it exists
 * for the same reason: with several figures piled on top of each other, a
 * control that acts on "the cutout" has to have exactly one of them in mind.
 */
export function activeLayerFor(i) {
    const f = store.frames[i];
    if (!f || !Array.isArray(f.layers)) return null;
    return f.layers[f.activeLayer] || null;
}

/**
 * Frame `i`'s title alignment, or null while it is still the channel's.
 *
 * Per frame and per title, like the halo and the gradient, because it is the
 * same kind of decision: the channel states the alignment its titles are set
 * to, and one particular title with one particularly short last line may want
 * another. Null rather than a copy of the channel's answer, so a frame that
 * has not been decided for follows its channel — including after the user
 * moves it to a different one.
 */
export function titleAlignFor(i) {
    const f = store.frames[i];
    return (f && f.titleAlign) || null;
}

/** Whether the frame's optional halo is switched on (see `glow.toggle`). */
export function glowFor(i) {
    const f = store.frames[i];
    return !!(f && f.glow);
}

/**
 * Which of its channel's backdrops frame `i` is shot on, as an index into
 * `background.textures` (see channels.js).
 *
 * Per frame, like the gradient and the flips, because it is the same kind of
 * decision: the channel owns the set of stages, and which one a given
 * thumbnail stands on is an answer about that one picture — a subject in a
 * dark jacket reads on one and disappears into the other.
 *
 * Wrapped by the caller rather than clamped here: the count belongs to the
 * channel, and a frame carried over to a channel with fewer backdrops than
 * the one it came from must land on a real one rather than on nothing.
 */
export function backgroundFor(i) {
    const f = store.frames[i];
    return (f && f.background) || 0;
}

/**
 * Which contour colour frame `i`'s cut-out figures are wearing, as an index
 * into `cutout.glow.colors` — or the length of that list, which is the step
 * meaning "no contour" (see `glow` in channels.js).
 *
 * One answer for every figure on the frame rather than one per figure: they
 * are the same subject shown two or three times, and a contour is what
 * separates that subject from the stage. Three of them in three colours would
 * be three different people.
 */
export function figureGlowFor(i) {
    const f = store.frames[i];
    return (f && f.figureGlow) || 0;
}

/**
 * Which of its channel's offered colours frame `i`'s title is set in, which
 * its picked words are set in, and which its slab is filled with — indices
 * into `title.colors.text`, `title.colors.highlight` and `title.colors.panel`
 * (see channels.js).
 *
 * Per frame, like the backdrop and the contour, and for the same reason: a
 * caption in white over a bright wall and the same caption in pink over a
 * dark one are two answers to what is in two photographs, not one answer
 * about the run.
 *
 * Three questions rather than one, because the pairing is the whole feature: a
 * channel offering the same three colours for the letters and for the slab
 * is offering nine combinations, and a single index would only ever give
 * three of them. The third is asked only by a channel whose picked words are
 * a colour the user chooses as well — see `colors.highlight` in channels.js,
 * and note that a channel NOT stating that list has its highlight recoloured
 * by `text` instead, which is what every palette here did until one wanted
 * the two to differ.
 *
 * Null means the channel decides, exactly as it does for titleAlignFor: the
 * pack's own `color` and `panel.fill` ARE its first answer, and a frame that
 * has not been decided for should wear them rather than a copy of them kept
 * somewhere else. Which is also what makes the palette's two lists free to be
 * in any order — the swatch that starts out marked is the one whose colour
 * matches what the channel already said, not whichever happens to be first
 * (see editor.renderColorButtons).
 *
 * Resolved by the caller rather than clamped here, exactly as backgroundFor
 * is — the list belongs to the channel, and an index carried over from one
 * with a longer palette falls back to the pack's own colour rather than to
 * nothing (see text.recolored).
 */
export function textColorFor(i) {
    const f = store.frames[i];
    return (f && f.textColor != null) ? f.textColor : null;
}

export function panelColorFor(i) {
    const f = store.frames[i];
    return (f && f.panelColor != null) ? f.panelColor : null;
}

export function highlightColorFor(i) {
    const f = store.frames[i];
    return (f && f.highlightColor != null) ? f.highlightColor : null;
}

/**
 * Which way frame `i`'s subject is cut out, as an index into
 * editor.CUTOUT_MODES.
 *
 * Per frame, and it has to be: the two methods do not rank. One is right for
 * a performer standing on a lit stage and the other for the frame where the
 * act's other half is not a person, and no property of the video says which
 * frame is which — only looking at it does.
 *
 * Wrapped by the caller rather than clamped here, exactly as backgroundFor is,
 * so the list can grow a third method without this needing to know.
 */
export function cutoutModeFor(i) {
    const f = store.frames[i];
    return (f && f.cutoutMode) || 0;
}

/**
 * Every image overlay on frame `i`, bottom of the stack first — array order
 * IS stacking order, so a newly added overlay lands on top of the ones
 * already there.
 */
export function logosFor(i) {
    const f = store.frames[i];
    return f && Array.isArray(f.logos) ? f.logos : [];
}

/**
 * The overlay the on-canvas handles belong to, or null.
 *
 * With one overlay per frame this question could not arise. With several, the
 * dashed box and resize handle have to belong to exactly one of them, or the
 * canvas fills with chrome and no drag has an unambiguous target.
 */
export function activeLogoFor(i) {
    const f = store.frames[i];
    if (!f || !Array.isArray(f.logos)) return null;
    return f.logos[f.activeLogo] || null;
}

// Flips and the gradient toggle are per-frame: each photo has its own subject
// orientation and contrast needs, and keeping the state on the frame means
// switching frames can never show one composed with another frame's flags.
// Sent together since they're all flags the backend's compose() reads in one
// shot.
export function framePresentation(i) {
    const f = store.frames[i];
    return {
        flip_image: !!(f && f.flipImage),
        flip_text:  !!(f && f.flipText),
        gradient:   !!(f && f.gradient),
    };
}

/**
 * The same three flags in the shape the BACKEND takes them, which differs
 * from the frame's own on any channel that has already decided what sits
 * between the photograph and the title: one that states a `scrim` draws its
 * gradient here, on the fabric canvas (see compose.addScrim), and one that
 * states a `photo` has shrunk the picture onto a ground of its own. Neither
 * wants the backend baking a second ramp into the pixels it returns.
 *
 * One button for one idea is the rule the Gradient button is written to (see
 * editor.toggleGradient), and this is what keeps it true on a channel that
 * has a scrim AND draws the photograph. The backend's gradient is a
 * full-height ramp down the side the title sits on, put there to clear space
 * for a headline beside a subject; a scrim is the channel's own shade, in the
 * channel's own direction. A frame that got both would be darkened twice, in
 * two directions, for one press.
 *
 * It was safe to leave out while the only channels with a scrim composed from
 * a cutout: they never draw the photograph, so what the backend did to it was
 * invisible. A capture channel does draw it, and there the second gradient is
 * the first thing anyone sees.
 *
 * The frame's own flag is left alone — this changes what is ASKED FOR, not
 * what the user chose. A frame carried between two channels keeps the answer
 * it was given, and gets the backend's ramp again the moment it is drawn in a
 * channel that has nothing of its own.
 */
export function backendPresentation(i) {
    const decor = decorFor(channelFor(i));
    const p = framePresentation(i);
    if (decor.scrim || decor.photo) p.gradient = false;
    return p;
}

/** The image currently representing a frame: its restored render if it has one. */
export function activeUrl(i) {
    const f = store.frames[i];
    if (!f) return null;
    return f.enhancedUrl || f.plainUrl;
}

/**
 * Replaces one of a frame's images, releasing the object URL it held.
 *
 * Every render the backend returns arrives as an object URL rather than a
 * base64 data string — a third fewer bytes on the wire and no encode/decode
 * pass on either side — which means the browser holds the blob alive until
 * it's explicitly revoked. Dropping the reference is not enough.
 */
export function setFrameImage(f, key, url) {
    if (f[key] && f[key] !== url) URL.revokeObjectURL(f[key]);
    f[key] = url;
}

/**
 * Drops a frame's cutout, releasing the object URL it held.
 *
 * The same obligation `setFrameImage` carries and for the same reason: the
 * cutout arrives as a blob URL, and the browser keeps the bytes alive until it
 * is revoked. A cutout is replaced on every reframe, zoom commit, preset
 * switch and variation, so leaking one per edit adds up quickly.
 */
/**
 * Drops every figure on a frame, releasing the object URLs they held.
 *
 * The same obligation `setFrameImage` carries and for the same reason: each
 * figure arrives as a blob URL, and the browser keeps the bytes alive until it
 * is revoked. Called where the photo underneath has been replaced — a
 * variation, an upload — so what those figures were pictures of is not there
 * any more.
 */
export function clearCutout(f) {
    if (!f) return;
    for (const layer of f.layers || []) {
        if (layer.src) URL.revokeObjectURL(layer.src);
    }
    f.layers = [];
    f.activeLayer = -1;
    f.cutoutKey = null;
    // The photo underneath is a different one, so the arrangement the user
    // made of the old one is not an arrangement of anything now.
    f.figuresRemoved = false;
}

export function releaseFrames() {
    for (const f of store.frames) {
        if (f.plainUrl) URL.revokeObjectURL(f.plainUrl);
        if (f.enhancedUrl) URL.revokeObjectURL(f.enhancedUrl);
        clearCutout(f);
    }
    store.frames = [];
}

/**
 * A brand-new slot.
 *
 * `moment` is what a capture run knows about WHEN this frame is — the seconds
 * into the source video, and the same instant as a timecode. Null for a
 * thumbnail run, which has no such thing, and never drawn anywhere in either
 * case: see the note on the fields themselves.
 *
 * `caption` is what was being SAID at that moment, for a capture channel that
 * asks for one (see backend/transcriber.py). It is not the title: it is where
 * the title starts from, and capture.applyCaptions is what turns it into
 * lines once there is a face to measure them against.
 */
export function makeFrame(frame_id, plainUrl, moment = null, caption = "") {
    return {
        frame_id, plainUrl, enhancedUrl: null,
        // Where in the video this frame came from — "frame 4 appeared at
        // 00m12s34". Recorded, never shown: nothing on screen reads either of
        // these, and nothing should start to without that being a deliberate
        // decision. They are here so the answer exists at all, for the
        // features that will want it.
        timestamp: moment ? moment.timestamp : null,
        timecode: moment ? moment.timecode : null,
        // The sentence the backend matched to that moment, kept whole and
        // unbroken. The lines below are what gets DRAWN and the user rewrites
        // them freely; this stays as it arrived, so the frame can always say
        // what was actually said over it.
        caption: caption || "",
        // One slot per line field the markup has (config.LINE_IDS); how many
        // of them a channel draws is the channel's (see linesFor).
        lines: ["", "", "", ""], hl: [0],   // each frame carries its own title
        // ...and its own channel: the brand a frame was made under stays with
        // it, so trying another one on a later frame cannot re-brand the ones
        // already finished. Empty until a channel is picked, which is what
        // stops a title being drawn in no brand at all.
        channel: activeChannelId(),
        // As with the gradient and the flips: set once the user picks a
        // channel while looking at THIS frame, and it stops a channel picked
        // afterwards on some other frame from reaching in and re-branding it.
        channelTouched: false,
        // Where the user has dragged and resized the title, for a channel
        // whose titles are placed by hand (`movable`). Null means untouched —
        // the block sits exactly where the channel's layout puts it.
        titleBox: null,
        titleActive: false,             // whether the title wears the canvas handles
        // The optional halo, for a channel that offers one (`glow.toggle`).
        // Off to begin with: it is an effect the user reaches for on a frame
        // that needs it, not a look every frame arrives wearing.
        glow: false,
        // Which of the channel's backdrops this frame stands on, and which
        // contour colour its figures wear (see backgroundFor/figureGlowFor).
        // Both start at 0, which is the channel's own first answer — the
        // backdrop the pack lists first, and, for the contour, the first
        // colour it offers. A channel that offers neither never reads them.
        background: 0,
        figureGlow: 0,
        // Which of the colours its channel offers this frame's title is set
        // in, which its picked words are set in, and which its slab is filled
        // with (see textColorFor). Null, unlike the two above, because a
        // palette is not a list the channel has no opinion about: the pack
        // states a `color`, a `highlight.color` and a `panel.fill` of its own,
        // so "not decided" already has an answer and index 0 would be a second
        // copy of it free to disagree.
        textColor: null,
        panelColor: null,
        highlightColor: null,
        logos: [],                      // and its own image overlays, bottom first
        activeLogo: -1,                 // which of them the canvas handles belong to
        // The subject cut out of THIS frame's photo, for a channel that
        // composes its thumbnails that way (see cutoutFor). Null until the
        // backend has produced one, and null again the moment the photo
        // changes — it is a picture of what is in this slot, so it cannot
        // outlive it.
        // The cut-out figures this frame shows, bottom of the stack first,
        // and which of them the arrows belong to. Empty until they arrive; how
        // many there are is the channel's (see `copiesBySlot`).
        layers: [],
        activeLayer: -1,
        // Which way this frame's subject is cut out of its photo — an index
        // into editor.CUTOUT_MODES (see cutoutModeFor).
        //
        // Method 2, and there is no button: the choice was offered for a while,
        // and then the reason it existed turned out to be somewhere else.
        //
        // The two methods differ in whether a person-segmentation prior gates
        // the saliency map, and on this app's own stage footage the gated one
        // measured cleaner — half the scenery, by a metric that turned out to
        // carry 4% of slack around the subject and therefore could not see the
        // thing users actually complain about, which is background wedged
        // AGAINST the body. What was really wrong was upstream of both: the
        // subtitle remover was reading the set's neon signage as captions and
        // inpainting a third of the picture, performer included. With that off
        // for this channel (see `cleanup` in the pack), the frames the choice
        // was being made on are different frames.
        //
        // The field stays rather than being folded back into a constant,
        // because the request and the cutout key both carry it and because a
        // channel that wants the other behaviour should be able to say so
        // rather than have it hard-coded at the call site. That is the same
        // reasoning that kept it here the first time.
        cutoutMode: 1,
        // What those figures were cut from — the canvas generation and the
        // edit preset, as one string. How editor.ensureCutout knows whether
        // they still describe this frame.
        cutoutKey: null,
        // Set once the user takes a figure off this thumbnail. It stops the
        // channel's own count putting it back: from then on how many figures
        // this frame shows is the user's answer, not the slot's.
        figuresRemoved: false,
        // Which way this title is set, when the user has overruled the
        // channel. Null means the channel decides — see titleAlignFor.
        titleAlign: null,
        editPreset: DEFAULT_PRESET,     // and its own edit preset
        // The gradient starts wherever the selected channel's branding puts
        // it; `gradientTouched` records that the user has since decided for
        // themselves, which stops a later channel switch from overruling them
        // (see editor.applyBrandingDefaults).
        gradient: branding().gradient,
        gradientTouched: false,
        // Same arrangement as the gradient, and for the same reason: the side
        // the title and the subject take is the channel's look, until the
        // user says otherwise on this frame. `flipsTouched` is that veto —
        // both flips share one, because they are one decision ("which way
        // round does this frame sit") made with two buttons.
        flipImage: branding().flipImage,
        flipText: branding().flipText,
        flipsTouched: false,
        // Set once the user replaces this slot's photo with their own. Not
        // carried over to other frames: it describes which picture this one
        // holds, which is the one thing a carry-over must never copy.
        uploaded: false,
        picked: false,
        edited: false,                  // flips true on the user's first real edit — see carryOverSettings
        version: 0,
    };
}

/**
 * Which frame a brand-new one takes its CONTENT from: the nearest worked-on
 * frame before it, and only failing that the nearest one after.
 *
 * "The frame being left behind" was the old answer, and it is the same answer
 * nearly always — frames are worked through left to right, so the one you
 * came from is the one before. It differs when the user jumps backwards into
 * a gap: standing on frame 7 and opening frame 3, what frame 3 should look
 * like is what frame 2 looks like, not what frame 7 does. Reading the
 * neighbours rather than the history also makes this independent of how the
 * user got here, which is one less thing for a caller to get wrong.
 *
 * Worked-on, because content is the one thing an untouched frame cannot give:
 * copying its blank title over a title typed two frames ago would be worse
 * than reaching past it.
 */
function contentSource(i) {
    // A frame with something to GIVE, before a frame that is merely flagged.
    //
    // `edited` was the test on its own, and it stopped meaning what this
    // function needs it to mean. It is set by everything the user can do to a
    // frame — the contour, the backdrop, the gradient, a zoom, a flip — and
    // none of those puts a word in the title. So a frame the user had only
    // recoloured became the nearest "worked-on" neighbour, and every blank
    // frame beside it inherited its blank title while a real one sat two
    // frames further along. Seen on a live session: frames 1 and 4 flagged and
    // empty, and frame 3 reaching back to frame 1 for nothing.
    //
    // Asking for content directly says what the docstring above always said it
    // meant. The `edited` scan stays as the fallback, because the other things
    // carried over — the edit preset above all — are worth taking from a
    // worked-on neighbour even when nobody has typed anything yet.
    const has = (f) => f && ((f.lines && f.lines.some(Boolean)) || (f.logos && f.logos.length));
    for (let k = i - 1; k >= 0; k--) if (has(store.frames[k])) return store.frames[k];
    for (let k = i + 1; k < store.frames.length; k++) if (has(store.frames[k])) return store.frames[k];
    for (let k = i - 1; k >= 0; k--) if (store.frames[k] && store.frames[k].edited) return store.frames[k];
    for (let k = i + 1; k < store.frames.length; k++) if (store.frames[k] && store.frames[k].edited) return store.frames[k];
    return null;
}

/**
 * Which frame a brand-new one takes its CHANNEL from: the one immediately
 * before it, whatever state that frame is in.
 *
 * A different rule from the content's, because a channel is a different kind
 * of thing. Every frame has one from the moment it exists, so there is no
 * such thing as reaching a neighbour with none to give — and stepping past
 * that neighbour to a "worked-on" frame further back is exactly how a frame
 * came out branded like frame 1 while the frame beside it was on something
 * else. A brand belongs to where you are in the strip, not to whoever last
 * typed something.
 */
function channelSource(i) {
    return store.frames[i - 1] || contentSource(i);
}

/**
 * Whether this frame's words are ITS OWN rather than something typed once and
 * worth spreading.
 *
 * The whole carry-over below rests on an assumption that holds for every
 * channel that makes thumbnails and fails completely for one that captions
 * stills: that a title is work the user did, so the next frame is better off
 * starting from it than from nothing. A caption is not work the user did. It
 * is what was being said over THIS frame — a different, already-correct
 * answer on every slot in the grid (see backend/transcriber.py).
 *
 * Carrying it over is therefore not a head start, it is data loss: the moment
 * the user clicked a frame, that frame's own sentence was replaced by its
 * neighbour's, which is exactly what was reported. The grid was right until
 * it was touched.
 *
 * Asked of the FRAME's channel rather than the page's, like everything else
 * here: a frame carries its brand with it.
 */
const wordsBelongToFrame = (f) => {
    const capture = captureFor((f && f.channel) || "");
    return !!(capture && capture.captions);
};

/**
 * A frame the user hasn't touched yet starts instead from whatever its
 * nearest worked-on neighbour has: channel, title, highlight, edit preset,
 * gradient, which way round it sits, and image overlays.
 *
 * It's a one-time carry-over, not a standing sync — once copied, this frame
 * is marked edited too, so it won't get silently overwritten again if the
 * source frame changes later or the user revisits this one. That is also what
 * keeps a channel switch made later from reaching it: from here on this frame
 * answers for itself (see editor.applyBrandingDefaults).
 *
 * Almost none of it applies to a channel whose frames arrive already carrying
 * their own words — see wordsBelongToFrame, and the short path below. There,
 * an image overlay is the one thing a user places by hand and would place the
 * same way twice, so it is the one thing that still travels.
 */
export function carryOverSettings(i, onPresetChanged) {
    const to = store.frames[i];
    if (!to || to.edited) return;

    // The channel first, and on its own terms: it comes from the frame beside
    // this one and asks nothing of it, so a frame picks up its neighbour's
    // brand even when neither of them has been typed on yet.
    const brandFrom = channelSource(i);
    if (brandFrom && brandFrom !== to) {
        to.channel = brandFrom.channel;
        to.channelTouched = brandFrom.channelTouched;
    }

    // Nothing further to carry while every frame is still at its defaults —
    // without this, merely clicking through untouched frames would lock each
    // one in as "edited" with blank content, blocking it from ever picking up
    // a title typed on an earlier frame afterwards.
    const from = contentSource(i);
    if (!from || to === from) return;

    // The short path: this frame already has the only text that is right for
    // it, so nothing about the title travels — not the words, not which of
    // them are picked out (a word index into somebody else's sentence), not
    // where the block was dragged to get it off a face that is not in this
    // photograph. The overlay does, for the reason given above.
    //
    // Marked edited all the same, so this stays the one-time decision the
    // rest of the function is.
    if (wordsBelongToFrame(to)) {
        to.logos = (from.logos || []).map(l => ({ ...l }));
        to.activeLogo = from.activeLogo;
        // The colours travel too, and they are the one thing here besides the
        // overlay that does. Everything the short path withholds is withheld
        // because it describes THESE words — which of them are picked out,
        // where the block had to be dragged to clear a face. A colour
        // describes none of that: it is the answer to "what does this run
        // look like", chosen once and meant for the set, and a grid where
        // every frame but the first reverted to the pack's first swatch would
        // be forty presses of the same button.
        to.textColor = from.textColor;
        to.panelColor = from.panelColor;
        to.highlightColor = from.highlightColor;
        to.edited = true;
        return;
    }

    to.lines = [...from.lines];
    to.hl = [...from.hl];
    // The title's placement and its halo ride along with the words they were
    // chosen for: a block moved off the subject's face on one frame was moved
    // there because of what the title says, and a new frame inheriting the
    // words without the position would put them back over the face.
    to.titleBox = from.titleBox ? { ...from.titleBox } : null;
    to.glow = from.glow;
    // The stage and the contour ride along for the reason the halo does: a
    // user who moved this video onto the second backdrop, or put a gold
    // contour on their subject, has made a decision about the VIDEO, and a
    // grid where every other thumbnail quietly reverts to the channel's first
    // answer is one they would have to make twenty times.
    to.background = from.background;
    to.figureGlow = from.figureGlow;
    // ...and the title's own colours, which are the same kind of decision
    // one step closer to the type: the user picked them looking at this
    // video, not at this frame.
    to.textColor = from.textColor;
    to.panelColor = from.panelColor;
    to.highlightColor = from.highlightColor;
    to.titleAlign = from.titleAlign;
    // The cut-out method rides along for the same reason, and it is the one of
    // these the user is most likely to have had to hunt for: a stage that
    // defeats the first method defeats it on every frame shot on that stage,
    // so making them find the second one twenty times would be the whole point
    // of the button, twenty times over. The FIGURES are still not inherited —
    // see the note below — only the answer about how to cut them.
    to.cutoutMode = from.cutoutMode;
    if (to.editPreset !== from.editPreset) {
        to.editPreset = from.editPreset;
        if (onPresetChanged) onPresetChanged(to);
    }
    // Copied one level deep: each overlay is its own record on the receiving
    // frame, so moving or deleting one there cannot reach back into the frame
    // it came from. The decoded image rides along by reference, which is the
    // point — it is the same picture and decoding it twice would be waste.
    to.logos = (from.logos || []).map(l => ({ ...l }));
    to.activeLogo = from.activeLogo;
    // The cutout is deliberately NOT among them. Everything else here is a
    // decision about how a thumbnail should look, which is worth inheriting;
    // a cutout is a picture of the person in ONE photo, and copying it onto a
    // frame showing a different moment would put the wrong person's wrong
    // expression on it. The new frame asks for its own — see editor.ensureCutout.
    //
    // The RULE it is cut under does ride along, and is the one part of the
    // cutout that should: whether the subject comes with what they are holding
    // is a fact about the ACT, and a ventriloquist is a ventriloquist on all
    // twenty frames. Having to press the button again on each of them would be
    // the tool asking the user to re-state something it has already been told.
    to.gradient = from.gradient;
    to.gradientTouched = from.gradientTouched;
    // The flips ride along with everything else. They were the one thing left
    // out, and it showed the moment a channel stopped deciding them for every
    // frame at once: a new frame inherited the title and the gradient of the
    // frame beside it and then sat the other way round from it.
    to.flipImage = from.flipImage;
    to.flipText = from.flipText;
    to.flipsTouched = from.flipsTouched;
    to.edited = true;
}
