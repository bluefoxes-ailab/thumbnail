import { CW, CH, LINE_IDS, BLOCK_ID, TITLE_FIELD_IDS, FIDELITY, DEFAULT_PRESET, el } from "./config.js";
import {
    frames, current, selectedIndex, linesFor, hlFor, logosFor,
    backendPresentation, setFrameImage, clearCutout, channelFor,
    titleAlignFor, layersFor, activeLayerFor, backgroundFor, figureGlowFor, cutoutModeFor,
    textColorFor, panelColorFor, highlightColorFor,
} from "./state.js";
import {
    channels, activeChannelId, setActiveChannel, hasChannel, titleStyle, branding,
    decorFor, usesCutout, NO_CHANNEL,
} from "./channels.js";
import { prepareTitleFont, prepareTitleTexture, wordsOf, wordKey } from "./text.js";
import { reportMissingTitleFont } from "./fontguard.js";
import { postImage, loadImage, discardImage, postImageFile } from "./net.js";
import { updatePreview, refreshThumb, refreshAllThumbs, fitCutout } from "./compose.js";
import {
    status, enhancing, showSpinner, maybeHideSpinner, syncBatchButton, markFrameReady,
} from "./ui.js";
import { invalidate as invalidateWide, prefetchWide } from "./wide.js";
import { syncZoomSliderFor } from "./reframe.js";
import { captureMode } from "./capture.js";

// ── Channel ───────────────────────────────────────────────────────────────

/**
 * Which channel the titles on this page belong to, and the gate in front of
 * the title fields.
 *
 * The fields are locked until a channel is chosen because there is no such
 * thing as an unbranded title here: the font, the colors, the highlight and
 * the layout all come from the channel (see channels.js), so typing before
 * picking one would either produce a title in some arbitrary house style or
 * one the renderer refuses to draw. Locking is the honest version of that,
 * as long as it says why — which is what the message below is for.
 */

const CHANNEL_REQUIRED_MSG = "Select a channel first";

let _warnTimer = null;

// The two places a channel can be chosen. They are one selection shown twice,
// not two settings: the one above the link box is reachable before a run
// starts (which is the only moment a channel can still change how the frames
// are CHOSEN — see `framing` in channels.js), and the one in the title panel is
// reachable afterwards, when there is a frame in front of the user to re-brand.
const CHANNEL_PICKERS = ["runChannelSelect", "channelSelect"];

/**
 * Fills both dropdowns from the registry, so adding a channel is a data change
 * only.
 *
 * Grouped under the label each channel belongs to (`group` in its pack), which
 * matters because the list is not one flat set of brands: it is several
 * networks' worth, and a user looking for one of them was reading seven names
 * in a column to find out which. The headings say where each run starts.
 *
 * Drawn as real `<optgroup>`s rather than as unselectable `<option>`s dressed
 * up to look like headings. That is not a shortcut, it is the only version
 * that is actually inert: a disabled option is still a row the arrow keys walk
 * onto and screen readers announce as a choice, where an optgroup label cannot
 * be landed on by any route — keyboard, mouse or assistive — because the
 * platform's own list widget knows it is not a choice. It also means the OS
 * draws it in whatever a heading looks like on that OS, which is one less
 * thing pretending to be native.
 *
 * A group opens when the label CHANGES as the ordered list is walked, so the
 * order the packs declare is the only thing deciding what sits under what — a
 * channel joins a heading by being ordered next to its stablemates, and there
 * is no second list of memberships to fall out of step with the first. A pack
 * with no label goes in loose.
 */
export function initChannelPicker() {
    for (const id of CHANNEL_PICKERS) {
        const sel = el(id);
        if (!sel) continue;
        // The blank "-" is first and is the default: the app opens assuming no
        // brand rather than the first one on the list. Outside every group,
        // because "no channel" belongs to no network.
        sel.innerHTML = "";
        const blank = document.createElement("option");
        blank.value = "";
        blank.textContent = "-";
        sel.appendChild(blank);

        let group = null, into = sel;
        for (const c of channels()) {
            if ((c.group || "") !== group) {
                group = c.group || "";
                if (group) {
                    into = document.createElement("optgroup");
                    into.label = group;
                    sel.appendChild(into);
                } else {
                    into = sel;
                }
            }
            const o = document.createElement("option");
            o.value = c.id;
            o.textContent = c.name;
            into.appendChild(o);
        }
    }
    syncChannelPickers();
    syncChannelFields();
}

/** Points both dropdowns at whatever is selected, wherever the change came from. */
export function syncChannelPickers() {
    for (const id of CHANNEL_PICKERS) {
        const sel = el(id);
        if (sel) sel.value = activeChannelId();
    }
}

/**
 * READONLY, not disabled, while no channel is selected.
 *
 * A disabled field takes no focus and fires no key events, so an attempt to
 * type into it does nothing at all — the user is left to guess why the app
 * ignored them. Readonly refuses the edit just as firmly but still reports
 * the keystroke, which is what lets the attempt be answered with the reason.
 */
export function syncChannelFields() {
    const unlocked = hasChannel();
    for (const id of TITLE_FIELD_IDS) {
        const input = el(id);
        input.readOnly = !unlocked;
        input.classList.toggle("locked", !unlocked);
        input.title = unlocked ? "" : CHANNEL_REQUIRED_MSG;
    }
    syncTitlePanelMode();
    if (unlocked) hideChannelWarning();
}

/**
 * A frame's stored lines as one block of text — the same trim the renderer
 * applies, so what the field shows is what would be drawn.
 */
function blockText(vals) {
    const lines = vals.map(l => l || "");
    while (lines.length && !lines[0].trim()) lines.shift();
    while (lines.length && !lines[lines.length - 1].trim()) lines.pop();
    return lines.join("\n");
}

/** True when the selected channel's titles are typed as one block. */
const freeformTitle = () => hasChannel() && !!titleStyle().layout.freeform;

/**
 * The line fields this channel actually uses, in order.
 *
 * Capped by what the markup has rather than by what a pack asks for: a channel
 * declaring five lines gets four, which is a title one line shorter than
 * intended — where honouring it would give a fifth line that can be stored, is
 * drawn on the canvas, and has no field anywhere to type it into or clear it
 * from. Everything that walks the slots walks THIS, so the panel, the
 * highlight buttons and what gets stored can never disagree about how many
 * there are.
 */
function lineIds() {
    if (!hasChannel()) return LINE_IDS.slice(0, 3);
    const wanted = titleStyle().layout.lines;
    return LINE_IDS.slice(0, Math.max(1, Math.min(wanted || LINE_IDS.length, LINE_IDS.length)));
}

/**
 * True when a highlight covers one word rather than a whole line.
 *
 * The scope alone, and not the mode with it. It used to ask for a boxed
 * highlight as well, because word scope was a box-only feature; it is not any
 * more (see `scope` in js/channels.js), and a panel that went on offering
 * lines to a channel that highlights words would be a panel whose buttons
 * pick out something other than what the canvas draws.
 */
function wordHighlight() {
    if (!hasChannel()) return false;
    const h = titleStyle().highlight;
    return !!h && h.scope === "word";
}

/**
 * Shows the parts of the title panel this channel actually has.
 *
 * Three things vary, and all three are the channel's: which field the title is
 * typed into, whether the optional halo has a button, and whether the block
 * can be moved on the canvas. Everything not offered is hidden rather than
 * disabled — unlike the highlight buttons, which go dead in place because
 * they are a feature every channel is expected to have and this one has
 * deliberately dropped. A glow button on a channel with no glow would not be
 * saying anything of the sort; it would just be a button that does nothing.
 */
export function syncTitlePanelMode() {
    const style = titleStyle();
    const freeform = freeformTitle();
    const used = lineIds();
    for (const id of LINE_IDS) el(id).classList.toggle("hidden", freeform || !used.includes(id));
    el(BLOCK_ID).classList.toggle("hidden", !freeform);

    if (freeform) autoGrowBlock();   // scrollHeight is 0 while it is hidden, so never before

    const offersGlow = hasChannel() && !!style.glow && !!style.glow.toggle;
    el("glowRow").classList.toggle("hidden", !offersGlow);
    el("titleMoveHint").classList.toggle("hidden", !(hasChannel() && style.movable));
    const cutout = usesCutout(activeChannelId());
    el("cutoutHint").classList.toggle("hidden", !cutout);
    // ...and on the body, because one of the controls a cutout channel needs
    // is hidden by a rule about the RUN rather than about the channel: a
    // frame fetch takes "Flip image" away, on the reasoning that there is no
    // title for the picture to sit beside and nothing to mirror the still
    // for. That reasoning stopped being complete the moment a capture channel
    // could compose from a cutout, because on one of those the button means
    // something else entirely — it mirrors the FIGURE (see toggleFlip), which
    // is an edit somebody makes on any channel that has figures at all.
    // The stylesheet reads this to tell the two cases apart.
    document.body.classList.toggle("cutout-channel", cutout);
    syncAlignButtons();
    renderColorButtons();   // ...and the swatches, which are the channel's too
    // The reframe hint promises a drag this channel does not offer: there is
    // no photograph on screen to reframe (see the press handler in reframe.js).
    el("reframeHint").classList.toggle("hidden", cutout);
    syncGlowButton();
}

/**
 * Sizes the title block to exactly the lines it holds.
 *
 * A textarea has a fixed height and scrolls; this one grows instead, starting
 * at a single row and taking a row for each break the user types. That is not
 * decoration — the field stands where three separate line fields stand for
 * every other channel, so a box that opens several rows deep reads as a title
 * that already has blank lines in it. This channel is the one where a blank
 * line is a real thing the user can type, which is precisely why the field
 * must not appear to contain any.
 *
 * `height: auto` first, and it is load-bearing: scrollHeight is the content
 * height OR the current height, whichever is larger, so measuring without
 * collapsing it first means the field can only ever grow.
 *
 * The border is added back because everything here is `border-box`, where
 * `height` includes the border while scrollHeight does not — without it the
 * field ends up 4px short of its own content and scrolls by a hairline.
 */
function autoGrowBlock() {
    const field = el(BLOCK_ID);
    if (!field) return;
    field.style.height = "auto";
    // An EMPTY field is left at the height its one row gives it. Chrome
    // measures the PLACEHOLDER into scrollHeight when there is nothing else
    // to measure, so growing to fit it opened the field two rows deep the
    // moment the placeholder was long enough to wrap — the exact thing this
    // is here to prevent, and from text that is not even the user's. (The
    // placeholder is short enough to fit one row for the same reason: at two
    // rows the second half of it would be clipped.)
    if (!field.value) return;
    const border = field.offsetHeight - field.clientHeight;
    field.style.height = `${field.scrollHeight + border}px`;
}

/** Says why the fields are refusing input. Fades out on its own; re-flashes on a repeat attempt. */
export function warnChannelRequired() {
    const box = el("channelWarning");
    box.textContent = CHANNEL_REQUIRED_MSG;
    box.classList.remove("hidden");
    // Restart the flash even when the message is already up — otherwise a
    // second attempt is met with a message that has been sitting there since
    // the first, and reads as stale rather than as an answer.
    box.classList.remove("flash");
    void box.offsetWidth;   // forces the class removal to take effect before it goes back on
    box.classList.add("flash");
    clearTimeout(_warnTimer);
    _warnTimer = setTimeout(hideChannelWarning, 4000);
}

function hideChannelWarning() {
    clearTimeout(_warnTimer);
    el("channelWarning").classList.add("hidden");
}

/** True when the title fields are usable; otherwise says why and lets the caller bail. */
export function guardChannel() {
    if (hasChannel()) return true;
    warnChannelRequired();
    return false;
}

/**
 * Switches the guidelines every title on the page is set with.
 *
 * Rendering happens twice on purpose: once immediately, and again once the
 * channel's own face has actually loaded. text.js refuses to cache anything
 * measured before that point and drops what it has when the font lands, but a
 * canvas already painted keeps its pixels — so the second pass is what
 * replaces a title composed against fallback metrics.
 */
export async function selectChannel(id) {
    const switched = setActiveChannel(id);
    syncChannelPickers();

    // The bookkeeping happens whether or not the page's own selection moved.
    // Picking the channel a frame is already on looks like a no-op and is
    // not one: it is the user saying this frame's brand is decided, and
    // returning early on it left the frame still open to being re-branded by
    // a choice made two frames later.
    //
    // The choice is written onto the frames it is about — see reachedByChoice.
    // Everything downstream reads the brand off the frame, so this assignment
    // IS the channel switch as far as the page is concerned; the module-level
    // selection is only what the next frame will be drawn with.
    frames().forEach((f, i) => { if (reachedByChoice(f, i)) f.channel = activeChannelId(); });
    // ...and the frame in front of the user has now been given its brand by
    // hand. That is a decision about this frame, exactly like clicking its
    // gradient button, so it gets the same veto: a channel picked later while
    // looking at some other frame will not reach back and undo it. Without
    // this, a frame the user had branded but not yet typed on was still fair
    // game, and would silently change under them.
    if (current()) current().channelTouched = true;
    if (!switched) return;   // nothing to redraw: same brand, same pixels

    syncChannelFields();
    // A channel that composes from a cutout needs one for the frame in front
    // of the user, and a channel that does not has no use for the one it may
    // be carrying. Both are this call, and it is fire-and-forget: the frame is
    // drawn now with whatever it has, and again when the cutout lands.
    //
    // The zoom control is resynced after it either way, because the switch may
    // have changed what that control MEANS — the subject's size on a cutout
    // channel, the crop window on every other one — and the two have different
    // bounds (see reframe.syncZoomSliderFor).
    ensureCutout(selectedIndex()).then(() => {
        const frame = current();
        if (frame) syncZoomSliderFor(frame.frame_id);
    });
    // The highlight buttons belong to the CHANNEL as much as to the frame:
    // a channel with no highlight renders them dead (see renderHlButtons).
    // Without this they keep whatever the previous channel left behind, so
    // leaving a channel that has no highlight took the feature away from
    // every channel picked after it until the frame changed.
    renderHlButtons();
    applyBrandingDefaults();
    updatePreview();
    refreshAllThumbs();

    const style = titleStyle();
    // Both of the channel's assets are fetched before it is drawn with, and
    // together: each render is synchronous and takes whatever has arrived, so
    // one landing a beat after the other would mean a pass composed with the
    // old font or a flat highlight, then a redraw.
    const [font, texture] = await Promise.all([
        prepareTitleFont(style),
        prepareTitleTexture(style),
    ]);
    // The face NAMED BY THE RESULT, not the channel's own: a channel may be
    // set in two — the block's and its highlight's — and a banner that always
    // said the block's would send the user looking for the wrong file (see
    // titleFaces in text.js).
    if (!font.ok) reportMissingTitleFont(font.face || style.face);
    // A missing texture is not the same class of problem as a missing face —
    // the layout is measured from the font, so a wrong one silently corrupts
    // every position, while a missing texture only leaves the highlight flat,
    // which is visible on sight and harms nothing else. It is logged rather
    // than put in front of the user as a broken install.
    if (!texture.ok) {
        console.warn(`Highlight texture ${style.highlight && style.highlight.texture} could not be ` +
                     "loaded — highlighted lines will be filled with the flat fallback colour.");
    }
    updatePreview();
    refreshAllThumbs();
}

/**
 * Points the page at the selected frame's own channel.
 *
 * The counterpart to selectChannel: that one is the user choosing a brand for
 * a frame, this one is the app catching up when the user moves to a frame
 * that already has one. Nothing is written to any frame here — the dropdown,
 * the lock on the title fields and the highlight buttons are just told what
 * this frame already is.
 *
 * Not awaited by its caller. The face may not be loaded if this frame's
 * channel is one the user has not visited in a while, and the two renders are
 * for the same reason selectChannel makes two: what is on screen now is drawn
 * against whatever metrics have arrived, and drawn again once the real ones
 * have.
 */
export async function syncChannelToFrame() {
    const f = current();
    const changed = setActiveChannel((f && f.channel) || NO_CHANNEL);
    syncChannelPickers();
    syncChannelFields();
    renderHlButtons();
    if (!changed || !hasChannel()) return;

    const style = titleStyle();
    const [font] = await Promise.all([prepareTitleFont(style), prepareTitleTexture(style)]);
    if (!font.ok) reportMissingTitleFont(font.face || style.face);
    updatePreview();
    refreshAllThumbs();
}

/**
 * Pushes the channel's non-type defaults onto the frames — the dark gradient,
 * and which side the title and the subject take.
 *
 * WHICH frames is the whole subtlety, and it used to be wrong. A channel is
 * chosen once for the page, but a channel switch is an edit made at a moment,
 * while looking at one particular frame. It reached every frame that had not
 * had these two specific controls clicked on it — which included every frame
 * the user had already worked on and moved past. Type a title on frame 1, go
 * to frame 2, change channel there, and frame 1 came back flipped or with its
 * gradient turned off: work the user had done, undone from another screen.
 *
 * So it now reaches exactly two kinds of frame:
 *
 *   - the one being looked at, because that is what the choice was about;
 *   - frames the user has never worked on, which have nothing to lose and
 *     everything to gain from arriving in the new brand's shape.
 *
 * A frame that has been edited and is not on screen is left alone, full stop.
 * It keeps what it was given until the user goes back to it and says
 * otherwise — see carryOverSettings for what a NEW frame inherits instead.
 *
 * Within a frame this reaches, the two "touched" flags still veto: an
 * explicit click on gradient or flip outranks a brand default even on the
 * frame in front of you. They are tracked apart because they are separate
 * decisions — deciding this frame keeps its gradient says nothing about which
 * way round it should sit.
 *
 * Frames not currently on screen need nothing further — every selection
 * re-composes a frame with its current flags — so only the previewed one is
 * recomposed here.
 */
/**
 * Which frames a channel choice made right now is allowed to reach: the one
 * being looked at, because that is what the choice was about, and any frame
 * that has neither been worked on nor been given a brand of its own, which
 * has nothing to lose by it.
 *
 * Everything else is left alone. Both the brand's presentation and the brand
 * itself are applied through this one predicate, so the two can never
 * disagree about which frames a channel switch touched.
 */
const reachedByChoice = (f, i) => i === selectedIndex() || (!f.edited && !f.channelTouched);

function applyBrandingDefaults() {
    if (!frames().length) return;
    const brand = branding();
    let currentChanged = false;
    frames().forEach((f, i) => {
        if (!reachedByChoice(f, i)) return;
        let changed = false;
        if (!f.gradientTouched && f.gradient !== brand.gradient) {
            f.gradient = brand.gradient;
            changed = true;
        }
        if (!f.flipsTouched) {
            if (f.flipImage !== brand.flipImage) { f.flipImage = brand.flipImage; changed = true; }
            if (f.flipText !== brand.flipText) { f.flipText = brand.flipText; changed = true; }
        }
        if (changed && i === selectedIndex()) currentChanged = true;
    });
    syncGradientButton();
    syncBackgroundButton();
    syncFigureGlowButton();
    syncFlipButtons();
    if (currentChanged) recompose({ previewFirst: false });
}

// ── Title panel ───────────────────────────────────────────────────────────

/**
 * The title fields are the source of truth for the SELECTED frame — the user
 * decides the line breaks, either by which of the three slots they fill in or
 * by where they press Enter in the block, depending on the channel (see
 * `freeform` in channels.js). Blank slots are dropped rather than rendered as
 * empty rows, so leaving one out just yields a shorter block; a blank line
 * inside a typed block is kept, because there it is a break the user made
 * (see state.linesFor).
 *
 * Stored as an array of lines either way, which is what lets a frame keep its
 * title while the user tries another brand on it: the two panels are two ways
 * of writing down the same thing.
 */
export function updateLines() {
    const f = current();
    if (!f) return;
    // Belt and braces: the fields are readonly without a channel, so this
    // should be unreachable from the keyboard.
    if (!hasChannel()) return;
    f.edited = true;
    f.lines = freeformTitle()
        ? el(BLOCK_ID).value.split("\n")
        // Every slot the markup has, not only the ones on screen: a line typed
        // under a four-line channel is still the user's when they try a
        // three-line one on the same frame, and it comes back if they switch
        // back. What is DRAWN is trimmed to the channel instead — see
        // state.linesFor.
        : LINE_IDS.map(id => el(id).value);
    f.hl = survivingHighlights();
    autoGrowBlock();   // a break just typed makes the field a row taller
    renderHlButtons();
    updatePreview();
    refreshThumb(selectedIndex());   // only this frame's title changed
}

/**
 * The highlights still standing after an edit — the ones naming something
 * that is still there.
 *
 * Only entries of the CURRENT channel's shape are judged. A frame carries one
 * set of highlights and two channels may pick things out differently (a line,
 * a word — see state.hlFor), so a word key sitting in the set while a
 * line-scope channel is selected is not stale, it is simply not this
 * channel's business: dropping it would quietly throw away the user's choices
 * the moment they looked at the frame under another brand.
 */
function survivingHighlights() {
    const lines = linesFor(selectedIndex());
    const hl = hlFor(selectedIndex());
    const isWordKey = (k) => typeof k === "string";
    if (!wordHighlight()) return hl.filter(k => isWordKey(k) || k < lines.length);
    return hl.filter((k) => {
        if (!isWordKey(k)) return true;
        const [li, wi] = k.split(":").map(Number);
        return lines[li] !== undefined && wordsOf(lines[li])[wi] !== undefined;
    });
}

/** Loads the selected frame's stored title into the side panel. */
export function loadTextPanel() {
    const f = current();
    const vals = (f && f.lines) || [];
    LINE_IDS.forEach((id, k) => { el(id).value = vals[k] || ""; });
    // Both fields are filled, whichever is on screen. A frame moved between a
    // three-slot channel and a block one then shows the title it already had
    // rather than an empty field, and the lines the user typed survive the
    // trip in both directions.
    //
    // Trimmed at both ends exactly as the renderer trims it (see
    // state.linesFor), which is what keeps an untouched frame's stored
    // ["", "", ""] from arriving as two empty lines the user then has to
    // delete. Blank lines BETWEEN two others are the user's own and stay.
    el(BLOCK_ID).value = blockText(vals);
    autoGrowBlock();
    syncChannelFields();   // the lock is the channel's, not the frame's, but the panel is redrawn per frame
    // What the box holds, named for what it is. A captured frame's text
    // arrived as the sentence somebody said over it and is edited as a
    // caption, not written as a headline, and calling it a title would send
    // the user looking for a title they never wrote (see capture.js).
    const capture = captureMode();
    const kind = capture && capture.captions ? "Caption" : "Title";
    el("textPanelLabel").textContent = `${kind} — Frame ${selectedIndex() + 1}`;
    renderHlButtons();
    syncAlignButtons();  // ...as does the alignment
    renderColorButtons();   // ...and the two colours, which are the frame's as well
    syncGlowButton();   // the halo is the frame's, so it follows the selection
    syncLogoControls();
    syncEditPresetButtons();
}

/**
 * All 4 buttons (Line 1/2/3, All) are always shown — a slot with no text just
 * renders disabled instead of disappearing, so the panel's layout stays put
 * as the user types instead of buttons popping in/out under the cursor. The
 * same goes for a channel that has no highlight at all (title.highlight null):
 * the buttons stay where they are and go dead, with a tooltip saying whose
 * decision that was. Live buttons that produced no visible change would read
 * as the app ignoring the click.
 * `hl`/addTextOverlay still index into the FILTERED non-empty list — a slot's
 * filtered index is however many non-empty slots come before it, computed
 * fresh here each render.
 *
 * `hl` holds a *set* of highlighted slot indices so lines toggle
 * independently and combine. "Highlight All" is derived, not a separate flag:
 * it shows active whenever every visible line is already in the set, and
 * clicking it either fills the set or empties it.
 */
export function renderHlButtons() {
    const c = el("hlButtons");
    c.innerHTML = "";
    const f = current();
    const rawLines = (f && f.lines) || [];
    const hl = hlFor(selectedIndex());
    const style = titleStyle();
    const highlightable = !hasChannel() || !!style.highlight;
    const noHighlightMsg = highlightable
        ? ""
        : "This channel has no highlight — every line is set the same way.";

    if (wordHighlight()) {
        renderWordButtons(c, hl, highlightable, noHighlightMsg);
        return;
    }
    c.className = "btn-row";

    // Over the slots this channel HAS, so the filtered indices the renderer
    // uses are counted the same way it counts them (see state.linesFor) — a
    // button numbered from all four slots would highlight the wrong line the
    // moment a channel used fewer.
    const used = lineIds();
    let filteredCount = 0;
    const slotIndex = used.map((_, k) => {
        const has = (rawLines[k] || "").trim().length > 0;
        const idx = has ? filteredCount : -1;
        if (has) filteredCount++;
        return idx;
    });

    const toggleLine = (idx) => {
        const frame = current();
        if (!frame) return;
        const cur = hlFor(selectedIndex());
        frame.hl = cur.includes(idx) ? cur.filter(v => v !== idx) : [...cur, idx];
        updateLines();
    };

    used.forEach((_, k) => {
        const idx = slotIndex[k];
        const has = idx !== -1;
        const b = document.createElement("button");
        b.className = "btn " + (has && hl.includes(idx) ? "btn-active" : "");
        b.textContent = `Highlight Line ${k + 1}`;
        b.disabled = !has || !highlightable;
        b.title = noHighlightMsg;
        b.onclick = () => toggleLine(idx);
        c.appendChild(b);
    });

    const allActive = filteredCount > 0 && hl.length === filteredCount;
    const all = document.createElement("button");
    all.className = "btn " + (allActive ? "btn-active" : "");
    all.textContent = "Highlight All";
    all.disabled = filteredCount === 0 || !highlightable;
    all.title = noHighlightMsg;
    all.onclick = () => {
        const frame = current();
        if (!frame) return;
        frame.hl = allActive ? [] : Array.from({ length: filteredCount }, (_, i) => i);
        updateLines();
    };
    c.appendChild(all);
}

/**
 * The title itself, as buttons — one per word, laid out the way the block is.
 *
 * This is the panel for a channel that highlights words rather than lines.
 * There is no list of slots to offer: the words ARE the choices, so the text
 * the user typed is what they click on, in the order and the lines they typed
 * it in. Which also means the panel reads as a rough proof of the block —
 * where the breaks fell, which word ended up where — before they look at the
 * canvas.
 *
 * Set in the case the canvas will set it in, for the same reason. A channel
 * that uppercases its titles is going to draw a box round PARIS, and a button
 * offering "paris" would be offering something slightly other than what
 * happens.
 *
 * The words are counted by the same function the renderer counts them with
 * (text.wordsOf), so a click cannot put a box round the word next to the one
 * that was clicked.
 */
function renderWordButtons(container, hl, highlightable, noHighlightMsg) {
    container.className = "btn-col";
    const style = titleStyle();
    const lines = linesFor(selectedIndex());
    const cased = (w) => (style.uppercase ? w.toUpperCase() : w);

    const keys = [];
    lines.forEach((line, li) => {
        const words = wordsOf(line);
        if (!words.length) return;   // a blank line the user typed — no words to offer
        const row = document.createElement("div");
        row.className = "btn-row word-row";
        words.forEach((word, wi) => {
            const key = wordKey(li, wi);
            keys.push(key);
            const b = document.createElement("button");
            b.className = "btn btn-word " + (hl.includes(key) ? "btn-active" : "");
            b.textContent = cased(word);
            b.disabled = !highlightable;
            b.title = noHighlightMsg;
            b.onclick = () => {
                const frame = current();
                if (!frame) return;
                const cur = hlFor(selectedIndex());
                frame.hl = cur.includes(key) ? cur.filter(v => v !== key) : [...cur, key];
                updateLines();
            };
            row.appendChild(b);
        });
        container.appendChild(row);
    });

    if (!keys.length) {
        const hint = document.createElement("div");
        hint.className = "panel-hint";
        hint.textContent = "Type a title, then click a word to highlight it.";
        container.appendChild(hint);
        return;
    }

    // Same "All" button the line panel carries, and derived the same way — it
    // is active exactly when nothing is left to pick.
    const allActive = keys.every(k => hl.includes(k));
    const all = document.createElement("button");
    all.className = "btn " + (allActive ? "btn-active" : "");
    all.textContent = "Highlight All";
    all.disabled = !highlightable;
    all.title = noHighlightMsg;
    all.onclick = () => {
        const frame = current();
        if (!frame) return;
        frame.hl = allActive ? [] : keys;
        updateLines();
    };
    container.appendChild(all);
}

// ── Which way the title is set ────────────────────────────────────────────

const ALIGNMENTS = [["alignLeftBtn", "left"], ["alignCenterBtn", "center"], ["alignRightBtn", "right"]];

/**
 * Whether this channel's block has any slack to distribute, and therefore
 * whether the alignment control can do anything at all.
 *
 * Only when the whole block is set at ONE size. Fitted line by line, every
 * line is stretched to the same width and there is nothing left over to put at
 * one end or the other — the control would be three buttons that visibly do
 * nothing, which is worse than three buttons that are not there. See
 * `uniformSize` and `align` in channels.js.
 */
const alignable = () => hasChannel() && titleStyle().layout.uniformSize !== false;

/**
 * Sets this frame's title alignment, or clears it back to the channel's.
 *
 * Per frame, like the gradient and the halo, and for the same reason: it is an
 * answer about one particular title. Clicking the alignment a frame is already
 * set to hands it back to the channel, so there is a way out of having decided
 * — the same shape of gesture as clicking a highlighted line to unhighlight it.
 */
export function setTitleAlign(align) {
    const f = current();
    if (!f) return;
    if (!guardChannel()) return;
    f.edited = true;
    f.titleAlign = titleAlignFor(selectedIndex()) === align ? null : align;
    syncAlignButtons();
    updatePreview();
    refreshThumb(selectedIndex());
}

export function syncAlignButtons() {
    const row = el("alignRow");
    if (!row) return;
    row.classList.toggle("hidden", !alignable());
    // The channel's own answer is what shows as active while the frame has not
    // been decided for, so the buttons say what the title IS rather than only
    // what the user has said about it.
    const current_ = titleAlignFor(selectedIndex())
        || (hasChannel() ? titleStyle().layout.align : null) || "center";
    for (const [id, name] of ALIGNMENTS) {
        el(id).className = "btn " + (current_ === name ? "btn-active" : "");
    }
}

// ── The colours a channel offers ──────────────────────────────────────────

/**
 * The palettes, as [the key in the pack, the row, its label, the frame's
 * field, how the channel's own answer is read out of its style].
 *
 * One table rather than three near-identical pairs of functions, because the
 * letters, the picked words and the slab are the same question asked three
 * times: which of these colours is this one, on this frame. Everything below
 * walks it, which is what the third row cost — it was written as a line here
 * rather than as a third copy of the rendering, exactly as this comment said
 * it would be when there were two.
 *
 * The rows are in the order the panel reads top to bottom, which is also the
 * order they sit on the type: the block, then the words picked out of it,
 * then the ground underneath.
 *
 * `own` is the last field because it is the one thing the three rows do not
 * share — where in the style the channel's own first answer lives. It is read
 * defensively for the same reason `paletteOf` guards below: a channel with no
 * highlight and no slab has neither object to reach into, and the row is
 * hidden rather than crashing on the way to hiding it.
 */
const PALETTES = [
    ["text", "textColorRow", "textColorLabel", "textColor",
        (style) => style.color],
    ["highlight", "highlightColorRow", "highlightColorLabel", "highlightColor",
        (style) => style.highlight && style.highlight.color],
    ["panel", "panelColorRow", "panelColorLabel", "panelColor",
        (style) => style.panel && style.panel.fill],
];

/**
 * The colours the selected channel offers for `which`, or null.
 *
 * Null rather than an empty list for three different situations that all mean
 * the same thing to the panel — no channel, no palette, a palette that names
 * this half and not the other — because the only thing the caller does with
 * the answer is decide whether the row exists.
 *
 * The slab's palette is also null on a channel with no slab, and the picked
 * words' on a channel with no highlight. Those buttons would recolour
 * something that is not drawn, which is the one kind of control this panel
 * does not have anywhere: a button that visibly does nothing reads as the app
 * ignoring the click (see the note on `alignable`).
 */
function paletteOf(which) {
    if (!hasChannel()) return null;
    const style = titleStyle();
    const list = (style.colors || {})[which];
    if (!Array.isArray(list) || !list.length) return null;
    if (which === "panel" && !style.panel) return null;
    if (which === "highlight" && !style.highlight) return null;
    return list;
}

/**
 * Draws every row of swatches the selected channel offers, and marks the ones
 * this frame is wearing.
 *
 * Built per render rather than once, for the reason the highlight buttons
 * are: how many there are and what colour each one is belong to the channel,
 * and the channel can be changed from this same panel while a frame is on
 * screen.
 *
 * A row with no palette behind it is hidden, its label with it, and the block
 * as a whole disappears when none of them is offered — this is a feature a
 * channel HAS or does not, unlike the highlight buttons, which stay in place
 * and go dead because every channel is expected to have one.
 */
export function renderColorButtons() {
    let any = false;
    for (const [which, rowId, labelId, field] of PALETTES) {
        const row = el(rowId), label = el(labelId);
        if (!row) continue;
        const colors = paletteOf(which);
        row.classList.toggle("hidden", !colors);
        if (label) label.classList.toggle("hidden", !colors);
        row.innerHTML = "";
        if (!colors) continue;
        any = true;

        // Marked, never unmarked. chosenColor answers -1 only for a pack that
        // contradicts itself — one whose own colour is not among the ones it
        // offers — and a row of colours with no ring on any of them reads as
        // a control the app has stopped listening to, which is a worse thing
        // to show the user than a ring on the wrong swatch that the first
        // press makes true. The press is still a real write in that case:
        // setTitleColor asks chosenColor, not this.
        const at = Math.max(chosenColor(which, colors), 0);
        colors.forEach((color, k) => {
            const b = document.createElement("button");
            b.type = "button";
            b.className = "btn-swatch " + (k === at ? "btn-active" : "");
            // Inline, because it IS the content of the button — a stylesheet
            // cannot know a colour that arrived in a pack.
            b.style.background = color;
            b.title = color;
            b.onclick = () => setTitleColor(which, k);
            row.appendChild(b);
        });
    }
    const block = el("colorRows");
    if (block) block.classList.toggle("hidden", !any);
}

/**
 * Which swatch of `which`'s row is the one on screen, as an index — or -1 for
 * none of them.
 *
 * Takes only the pack's key, and looks the rest up in PALETTES, because the
 * frame's field and the way the channel's own answer is read out of the style
 * are two facts about that same key: passing them alongside it invited a call
 * naming one row's field with another row's colours, which is a swatch marked
 * from the wrong question and looks exactly like a correct one.
 *
 * Three answers in order, and the second is the one that matters. A frame
 * nobody has pressed a swatch on holds null, which means the channel decides
 * (see state.textColorFor), so what shows as chosen is the swatch whose colour
 * IS the channel's — the buttons then say what the title is rather than only
 * what the user has said about it, which is the same thing syncAlignButtons
 * does with the alignment.
 *
 * That is also what frees the two lists to be written in any order: nothing
 * anywhere assumes the channel's own colour is the one listed first.
 *
 * -1 when a stored index has no colour behind it — a frame carried over from
 * a channel with a longer palette — or when the pack's own colour is not in
 * its own list. The renderer falls back to that colour in both cases (see
 * text.recolored), so no swatch is marked because none of them is what is
 * drawn, which is the truth rather than a tidier lie.
 */
function chosenColor(which, colors) {
    const row = PALETTES.find(([key]) => key === which);
    if (!row) return -1;
    const [, , , field, ownOf] = row;
    const stored = STORED_COLOR[field](selectedIndex());
    if (stored != null && stored < colors.length) return stored;

    // Everything else is the channel's own colour, because that is what the
    // renderer draws for it. text.recolored makes no distinction between the
    // frame nobody has pressed this row's swatch on and the frame carrying an
    // index with no colour behind it — both get the pack's `color`,
    // `highlight.color` or `panel.fill` — so neither is a distinction this
    // function may make either.
    //
    // The out-of-range index used to stop here with -1, and that was the one
    // way a whole row could end up with nothing marked on it: switch a run
    // from a six-colour channel to a three-colour one and every frame holding
    // index 3, 4 or 5 lost its ring, while the title on screen was plainly
    // one of the three offered. The swatch matching the pack's own colour is
    // both the honest answer and the one the user is looking for.
    const own = ownOf(titleStyle());
    if (!own) return -1;
    const same = (a, b) => String(a).trim().toLowerCase() === String(b).trim().toLowerCase();
    return colors.findIndex(c => same(c, own));
}

// What each row's index is stored under on the frame. A table rather than a
// chain of conditionals for the reason PALETTES itself is one: the rows differ
// in nothing but their four names, and a fourth row should be four strings and
// a reader, not a fourth branch in two functions.
const STORED_COLOR = {
    textColor: textColorFor,
    highlightColor: highlightColorFor,
    panelColor: panelColorFor,
};

/**
 * Sets one of this frame's two title colours.
 *
 * Per frame, like the gradient and the halo, and for the same reason: it is
 * an answer about one picture. Unlike the alignment there is no way back to
 * "the channel decides" — every entry in the palette is one of the channel's
 * own answers, so a second press on a swatch has nothing to release.
 *
 * Which is why a press on the swatch already showing can be a no-op — but
 * only when the frame has nothing to write. That is a question about what the
 * frame STORES, and the swatch on screen does not answer it, because two
 * different frames put the same ring in the same place:
 *
 *   stores null          the channel is deciding, and the ring is on the
 *                        swatch matching the channel's own colour. There is
 *                        nothing to write that would not be a copy of an
 *                        answer the pack already gives, and writing it would
 *                        mark the frame edited — an edited frame stops
 *                        inheriting from its neighbours (see
 *                        state.carryOverSettings), which is a large
 *                        consequence for a click that changed nothing.
 *
 *   stores an index      carried in from a channel with a longer palette, so
 *   with no colour       the ring is on the pack's own colour for the same
 *   behind it            reason (see chosenColor). This one LOOKS identical
 *                        and is not: leave it alone and the dead index stays
 *                        on the frame, ready to come back the moment the user
 *                        returns to the longer channel. The press is what
 *                        pins the colour the user can see, so it has to write.
 *                        The frame is already edited either way — an index
 *                        only ever gets there through this function or
 *                        through the carry-over, and both set the flag.
 */
export function setTitleColor(which, index) {
    const f = current();
    if (!f) return;
    if (!guardChannel()) return;
    const colors = paletteOf(which);
    if (!colors || index < 0 || index >= colors.length) return;
    const row = PALETTES.find(([key]) => key === which);
    if (!row) return;
    const field = row[3];
    const stored = STORED_COLOR[field](selectedIndex());
    if (stored === index) return;
    if (stored == null && chosenColor(which, colors) === index) return;
    f.edited = true;
    f[field] = index;
    renderColorButtons();
    updatePreview();
    refreshThumb(selectedIndex());
}

// ── The optional halo ─────────────────────────────────────────────────────

/**
 * Turns the channel's glow on or off for this frame.
 *
 * Per frame, like the gradient and the flips, and for the same reason: it is
 * an answer to what is in one photograph. A halo that lifts a title off a
 * busy background is a smear on an empty one.
 */
export function toggleGlow() {
    const f = current();
    if (!f) return;
    if (!guardChannel()) return;
    f.edited = true;
    f.glow = !f.glow;
    syncGlowButton();
    updatePreview();
    refreshThumb(selectedIndex());
}

export function syncGlowButton() {
    const btn = el("glowBtn");
    if (!btn) return;
    const f = current();
    const on = !!(f && f.glow);
    btn.className = "btn " + (on ? "btn-active" : "");
    btn.textContent = on ? "Glow On" : "Glow Off";
}

// ── Edit presets ──────────────────────────────────────────────────────────

export function syncEditPresetButtons() {
    const preset = (current() && current().editPreset) || DEFAULT_PRESET;
    for (const [id, name] of [["presetNoneBtn", "none"], ["presetNaturalBtn", "natural"], ["presetFullBtn", "full"]]) {
        el(id).className = "btn " + (preset === name ? "btn-active" : "");
    }
}

/**
 * Switching presets re-runs the editing pipeline for this frame at the new
 * intensity — same enhance path used on first selection, so it reuses its own
 * busy-state handling and applies to whatever crop/flip the frame is at.
 */
export function setEditPreset(preset) {
    const f = current();
    if (!f || f.editPreset === preset) return;
    f.edited = true;
    f.editPreset = preset;
    invalidateWide(f.frame_id);   // stale — held the previous preset's restored pixels
    syncEditPresetButtons();
    enhanceFrame(selectedIndex());
    // ...and the cutout, ASKED FOR HERE rather than left to the line above.
    //
    // enhanceFrame drops the call outright when a pass is already in flight
    // for this frame, and the cutout used to be requested only from inside
    // that pass's success path — so a preset clicked while the frame was still
    // being restored changed the photo (the dropped pass redoes itself from
    // its own `finally`) and silently never changed the subject standing on
    // the backdrop, which is the only thing the user can actually see on this
    // channel. The two are separate endpoints with separate guards, so this is
    // simply the other one being asked.
    ensureCutout(selectedIndex());
}

// ── Enhance ───────────────────────────────────────────────────────────────

export async function enhanceFrame(i) {
    const f = frames()[i];
    if (!f) return;
    if (enhancing.has(f.frame_id)) return;   // already in flight for this frame
    enhancing.add(f.frame_id);

    // Canvas generation and preset this pass belongs to — a reframe landing
    // while the request is in flight bumps the version, and a preset click
    // changes editPreset, either of which makes this result stale: it was
    // computed from the previous crop/pipeline.
    const version = f.version || 0;
    const preset = f.editPreset || DEFAULT_PRESET;

    // The backend restores a frame's whole pre-crop view at most once — near
    // instant on every call after that, but the very first select of a
    // never-before-seen frame pays that cost here.
    if (i === selectedIndex()) showSpinner();

    try {
        const res = await postImage("/enhance-frame", {
            frame_id: f.frame_id, fidelity: FIDELITY, preset,
            ...backendPresentation(i),
        });
        if (!res.ok) return;
        // Stale — the finally below redoes it. The image that came back is of
        // the previous crop/preset and will never be shown, so it goes now
        // rather than being left for the tab to hold onto.
        if ((f.version || 0) !== version) return discardImage(res);
        if ((f.editPreset || DEFAULT_PRESET) !== preset) return discardImage(res);
        // The response is always a fully composed, valid frame here —
        // res.enhanced only says whether face restoration itself ran (false
        // when no restorer backend is available; the preset no longer
        // affects it, since restoration runs at every preset), not whether
        // the request produced something to show. Always apply it.
        setFrameImage(f, "enhancedUrl", res.url);
        refreshThumb(i);
        // The card stops being grey here and not a line earlier: what makes it
        // ready is its finished pixels, and this is where they arrive (see
        // ui.markFrameReady).
        markFrameReady(i);
        if (selectedIndex() === i) updatePreview();
        // The figures are cut from this same photo at this same preset, so
        // this is a moment they can be asked for. Not the only one — see
        // setEditPreset and warmCutouts, both of which reach a frame this pass
        // may never run for.
        ensureCutout(i);
        // Warm the drag-ghost cache before the first drag needs it, and take
        // the frame's own zoom bounds from it once it lands — they're
        // per-frame (how far the automatic framing already zoomed decides how
        // much manual room is left), so until the view arrives the slider is
        // still showing the markup's placeholder range.
        prefetchWide(f.frame_id).then(() => {
            if (selectedIndex() === i) syncZoomSliderFor(f.frame_id);
        });
    } catch (_) {
        // network / server error — degrade silently
    } finally {
        enhancing.delete(f.frame_id);
        if ((f.version || 0) !== version || (f.editPreset || DEFAULT_PRESET) !== preset) {
            enhanceFrame(i);   // re-cropped or preset changed mid-flight — redo with current state
        } else {
            maybeHideSpinner();
        }
    }
}

/**
 * Resolves once frame `i` is carrying its restored pixels — running the
 * enhance pass if nothing has yet, and waiting out one already in flight if
 * something has.
 *
 * enhanceFrame is fire-and-forget by design: it is called on selection, it
 * refuses to start a second pass over a frame, and when a reframe or a preset
 * click lands mid-flight it re-runs ITSELF from its own `finally`. Awaiting it
 * therefore tells you very little — it can return instantly because another
 * pass is already going, and it can resolve while the redo it just started is
 * still working.
 *
 * So this waits on the fact rather than on the call. Polling, because the fact
 * is a Set that three different code paths add to and remove from (see
 * ui.enhancing), and the alternative is threading a promise through every one
 * of them to answer a question only the two download paths and the capture
 * warm-up ever ask. The interval is invisible next to what is being waited
 * for: a restoration pass is seconds, this is eighty milliseconds.
 */
const _settled = (f) => new Promise((resolve) => {
    const check = () => (enhancing.has(f.frame_id) ? setTimeout(check, 80) : resolve());
    check();
});

export async function ensureEnhanced(i) {
    const f = frames()[i];
    if (!f) return;
    await _settled(f);
    if (!f.enhancedUrl) await enhanceFrame(i);
    // Again afterwards: the pass above re-runs itself when the frame moved on
    // while it was out, and that redo is started from inside the call that has
    // just resolved.
    await _settled(f);
}

// ── Presentation toggles ──────────────────────────────────────────────────

/** The figure "Flip image" acts on: the selected one, or the front one. */
const flippableFigure = () =>
    activeLayerFor(selectedIndex()) || layersFor(selectedIndex()).slice(-1)[0] || null;

export function syncFlipButtons() {
    const f = current();
    // On a channel that composes from cutouts the button mirrors one FIGURE
    // and not the photograph (see toggleFlip), so what it has to report is
    // that figure's state — reading the frame's own flag there would leave it
    // permanently unlit while visibly doing something.
    const figure = flippableFigure();
    const mirrored = figure ? figure.flip : !!(f && f.flipImage);
    el("flipImageBtn").className = "btn " + (mirrored ? "btn-active" : "");
    el("flipTextBtn").className  = "btn " + (f && f.flipText  ? "btn-active" : "");
}

/**
 * The Gradient button, and whether this channel has one at all.
 *
 * A channel with a `scrim` always has one, because the scrim IS what the
 * button switches (see toggleGradient). Otherwise it is offered only where the
 * backend's gradient would land on something a viewer can see — which means a
 * channel that draws the frame's photograph edge to edge, and nothing else.
 *
 * Two kinds of channel fail that and are the reason this test exists. One
 * composes from a cutout and never draws the photograph at all: Laugh Society
 * 1 is the case, and the button sat there through every frame doing nothing
 * whatever, which is worse than a button that is not offered. The other
 * shrinks the photograph onto a ground of its own (see `photo` in
 * channels.js), where the backend's ramp would darken one side of a card the
 * pack deliberately sized and centred — not nothing, but not what this button
 * means anywhere else either, which is the same fault wearing a different
 * face. Hiding it is the answer this panel already gives for the halo and the
 * alignment.
 */
export function syncGradientButton() {
    const f = current();
    const on = !!(f && f.gradient);
    const btn = el("gradientBtn");
    const decor = decorFor(channelFor(selectedIndex()));
    const acts = !!decor.scrim || (!decor.cutout && !decor.photo);
    btn.className = ["btn", on ? "btn-active" : "", acts ? "" : "hidden"]
        .filter(Boolean).join(" ");
    btn.textContent = on ? "Gradient On" : "Gradient Off";
}

/**
 * Re-composes the selected frame after a presentation flag changed. The image
 * mirror and the gradient are baked into what the backend returns; only the
 * text layout is mirrored locally on the canvas, which is why the text side
 * can update instantly while the image follows.
 *
 * Shared by all three toggles — they differ only in which flag they flip and
 * whether a local preview refresh makes sense before the round trip.
 */
async function recompose({ previewFirst }) {
    const idx = selectedIndex();
    const f = frames()[idx];
    if (!f) return;
    if (previewFirst) updatePreview();

    showSpinner();
    try {
        const res = await postImage("/compose-frame", {
            frame_id: f.frame_id, ...backendPresentation(idx),
        });
        if (res.ok) {
            // /compose-frame reuses the cached restored canvas when there is
            // one, so an already-enhanced frame recomposes instantly with no
            // second enhance pass owed.
            if (res.enhanced) {
                setFrameImage(f, "enhancedUrl", res.url);
            } else {
                setFrameImage(f, "plainUrl", res.url);
                setFrameImage(f, "enhancedUrl", null);
            }
            refreshThumb(idx);
            updatePreview();
            if (!res.enhanced) enhanceFrame(idx);
        }
    } catch (_) { /* network error — the local flag already flipped; the next call retries */ }
    maybeHideSpinner();
}

export async function toggleFlip(key) {
    const f = current();
    if (!f) return status("No frame selected");

    // On a channel that composes from cutouts, "Flip image" means the FIGURE
    // and not the photograph — there is no photograph on screen to mirror, and
    // on a thumbnail carrying three of them it can only sensibly mean one. The
    // selected one, or the front one when nothing is selected, which is the
    // same figure the zoom control acts on.
    //
    // Done on the canvas, so it costs nothing: a mirrored figure is the same
    // cutout drawn the other way round (see compose.addLogoOverlay). Asking
    // the backend would mean restoring the photo again to produce pixels the
    // browser already holds.
    const figure = key === "flipImage" ? flippableFigure() : null;
    if (figure) {
        f.edited = true;
        figure.flip = !figure.flip;
        syncFlipButtons();
        updatePreview();
        refreshThumb(selectedIndex());
        return;
    }

    f.edited = true;
    f[key] = !f[key];
    f.flipsTouched = true;   // an explicit choice — brand defaults stop overwriting it
    syncFlipButtons();
    await recompose({ previewFirst: true });   // text side flips instantly; the image follows
}

export async function toggleGradient() {
    const f = current();
    if (!f) return status("No frame selected");
    f.edited = true;
    f.gradient = !f.gradient;
    f.gradientTouched = true;   // an explicit choice — brand defaults stop overwriting it
    syncGradientButton();

    // A channel with a scrim draws its gradient HERE, on the fabric canvas
    // (see compose.addScrim), so there is nothing to ask the backend for and
    // the redraw is immediate. Everywhere else the gradient is baked into the
    // composed photo and the button means a round trip.
    if (decorFor(channelFor(selectedIndex())).scrim) {
        updatePreview();
        refreshThumb(selectedIndex());
        return;
    }
    // No local preview to update first — the gradient isn't drawn on the
    // fabric canvas, only baked into the backend's composed photo.
    await recompose({ previewFirst: false });
}

// ── Variation ─────────────────────────────────────────────────────────────

/**
 * Swaps this frame's underlying photo for a nearby moment of the same shot
 * (random direction/distance, bounded server-side so it stays a variation,
 * not a new scene) — for when the subject blinked or the user just wants a
 * fresh expression without losing the framing/title work already done.
 */
/**
 * Walks ONE cut-out figure on to another moment of the video.
 *
 * What Variation means on a channel that composes from cutouts. The button's
 * promise — "a different moment of this same shot" — is unchanged; what moves
 * is the scope. A thumbnail carrying three figures is an arrangement the user
 * built, and swapping all three at once would throw it away to answer a
 * question they asked about one of them.
 *
 * The next moment is the next one no OTHER figure on this thumbnail is already
 * showing, so clicking through never lands on a duplicate of the figure
 * standing beside it. The moments wrap (see api.CutoutRequest.copy), so the
 * button keeps working however long the user keeps pressing it.
 */
async function varyFigure(i, f, figure) {
    const btn = el("varyBtn");
    btn.disabled = true;
    showSpinner();
    try {
        // Which moment to walk this figure to. The copies the OTHER figures
        // are holding are skipped, so a click moves this one somewhere the
        // thumbnail is not already showing.
        const total = Math.max(1, figure.moments || 1);
        const taken = new Set(f.layers.filter(l => l !== figure).map(l => l.copy));
        // ...but a step is taken even when every copy is spoken for, and that
        // is what makes this button work at all rather than most of the time.
        //
        // Two ways it used to come out at the figure's own copy and change
        // nothing. A figure at copy 0 is the frame's own photo, and the
        // backend only counts the alternates when it is asked for one of them
        // (see api, `moments = ... if copy else []`) — so `moments` arrives as
        // 0, `total` clamps to 1, and the only other candidate is the copy the
        // next figure along is already holding. And on a three-figure slot
        // where the video yielded two alternates, every copy in the cycle
        // belongs to some figure, so the search runs out with nothing free.
        //
        // Both left `next` at the value it started with, and the click became
        // a re-cut of the same instant: the picture never changed, on any of
        // the three. Advancing regardless can land on a copy another figure is
        // showing, which is a visible repeat the user can click past — where
        // a dead button is one they can only wonder about.
        let next = (figure.copy + 1) % (total + 1);
        for (let step = 1; step <= total + 1; step++) {
            const candidate = (figure.copy + step) % (total + 1);
            if (!taken.has(candidate) && candidate !== figure.copy) { next = candidate; break; }
        }

        const res = await postImage("/cutout-frame", {
            frame_id: f.frame_id, fidelity: FIDELITY,
            preset: f.editPreset || DEFAULT_PRESET, copy: next,
        });
        if (!res.ok) {
            status(res.detail || "No other moment of this frame could be used");
            return;
        }
        const img = await loadImage(res.url);
        // Held by identity, not by index: a re-cut landing while this was out
        // replaces every figure on the frame, and writing into the old one
        // would put a picture somewhere nothing is drawing from.
        if (!f.layers.includes(figure)) {
            URL.revokeObjectURL(res.url);
            return;
        }
        if (figure.src) URL.revokeObjectURL(figure.src);
        Object.assign(figure, {
            src: res.url, _img: img, _alpha: null,
            natW: img.width, natH: img.height,
            copy: (res.meta && res.meta.copy) || 0,
            moments: (res.meta && res.meta.moments) || total,
        });
        f.edited = true;
        if (selectedIndex() === i) updatePreview();
        refreshThumb(i);
    } catch (_) {
        status("Network error creating variation");
    } finally {
        btn.disabled = false;
        maybeHideSpinner();
    }
}

export async function varyFrame() {
    const idx = selectedIndex();
    const f = frames()[idx];
    if (!f) return status("No frame selected");

    // On a channel that composes from cutouts, Variation belongs to one
    // figure: the selected one, or the front one when nothing is selected —
    // the same figure the zoom control and the flip act on.
    const spec = decorFor(channelFor(idx)).cutout;
    const figure = spec ? (activeLayerFor(idx) || layersFor(idx).slice(-1)[0]) : null;
    if (figure) return varyFigure(idx, f, figure);

    const btn = el("varyBtn");
    btn.disabled = true;
    showSpinner();
    try {
        const res = await postImage("/vary-frame", { frame_id: f.frame_id, ...backendPresentation(idx) });
        if (!res.ok) {
            status(res.detail || "No nearby variation found for this frame");
            return;
        }
        invalidateWide(f.frame_id);   // belonged to the source photo just swapped out
        setFrameImage(f, "plainUrl", res.url);
        setFrameImage(f, "enhancedUrl", null);   // force a fresh enhance pass for the new source
        f.version = (f.version || 0) + 1;
        // The subject in it is the one that was just replaced. Dropped now
        // rather than left to be overwritten when the new one arrives, so the
        // frame never shows one video moment standing in front of another.
        clearCutout(f);
        if (selectedIndex() === idx) updatePreview();
        refreshThumb(idx);
        enhanceFrame(idx);
    } catch (_) {
        status("Network error creating variation");
    } finally {
        btn.disabled = false;
        maybeHideSpinner();
    }
}

/**
 * Replaces the selected slot's photo with an image the user picked, then puts
 * it through everything a slot built from the video goes through.
 *
 * Nothing here is a special case downstream. The backend swaps the photo the
 * slot points at and returns the same plain crop /vary-frame does, so the
 * enhance pass fired at the end applies the frame's current edit preset to
 * it, and every later reframe, zoom, flip and preset switch reaches it
 * through the endpoints they always used. The text and image overlays are
 * composed over the frame on the canvas, so they stay above it untouched.
 */
export async function handleFrameFile(file) {
    const idx = selectedIndex();
    const f = frames()[idx];
    if (!f) return status("No frame selected");

    const btn = el("frameUploadBtn");
    btn.disabled = true;
    showSpinner();
    try {
        const res = await postImageFile("/upload-frame", file, {
            frame_id: f.frame_id, ...backendPresentation(idx),
        });
        if (!res.ok) {
            status(res.detail || "That image could not be used as a frame");
            return;
        }
        // The wide view is a render of the photo just replaced, and the frame
        // is on a new one — same reasoning as a Variation swap.
        invalidateWide(f.frame_id);
        setFrameImage(f, "plainUrl", res.url);
        setFrameImage(f, "enhancedUrl", null);   // force a fresh pass over the new photo
        clearCutout(f);   // of the photo this upload just replaced — see varyFrame
        f.uploaded = true;
        f.edited = true;
        f.version = (f.version || 0) + 1;
        syncVaryButton();
        if (selectedIndex() === idx) updatePreview();
        refreshThumb(idx);
        enhanceFrame(idx);
        status("Frame replaced with your image");
    } catch (_) {
        status("Network error uploading the frame");
    } finally {
        btn.disabled = false;
        maybeHideSpinner();
    }
}

/**
 * Variation offers "a nearby moment of this same frame", which a slot showing
 * an uploaded still does not have — the backend refuses it for that reason.
 * Greying the button out says so before the click rather than after it.
 */
export function syncVaryButton() {
    const f = current();
    const btn = el("varyBtn");
    const uploaded = !!(f && f.uploaded);
    btn.disabled = uploaded;
    btn.title = uploaded
        ? "Unavailable on a frame you uploaded \u2014 there is no nearby moment of it to try"
        : "Try a nearby moment of this same frame \u2014 a different expression, without changing the scene";
}

// ── The cutout subject ────────────────────────────────────────────────────

/**
 * frame_id -> the request currently producing that frame's figures.
 *
 * Same job `enhancing` does for the restoration pass, and needed for the same
 * reason: selecting a frame, switching its preset and the background warm-up
 * all ask for the same figures, and without this the second of them would
 * start a second set of segmentations while the first was still running.
 *
 * The PROMISE is kept, not merely the id, so that a caller which genuinely has
 * to wait for the result — the batch download, which is about to composite the
 * frame — waits for the one already running instead of being told "busy" and
 * compositing a backdrop with nobody on it.
 */
const _cuttingOut = new Map();

/**
 * How many figures this grid slot shows.
 *
 * A property of the SLOT, not of the photo in it: the channel offers twenty
 * options and some of them are arrangements rather than single figures (see
 * `copiesBySlot` in js/channels.js), so which is which has to be the same
 * every run and has to be readable from the pack. A slot past the end of the
 * list gets the last entry, so a video that yields more frames than the list
 * describes still gets an answer.
 */
function copiesForSlot(spec, i) {
    const list = spec.copiesBySlot;
    if (!Array.isArray(list) || !list.length) return 1;
    const n = list[Math.min(i, list.length - 1)];
    return Math.max(1, Math.min(Number(n) || 1, 6));
}

/**
 * Makes sure frame `i` is showing the figures its channel asks for — fetching
 * them when what it holds is not a picture of this frame any more, and
 * dropping them when its channel has no use for a cutout at all.
 *
 * Fire-and-forget from nearly every caller: the frame is drawn now with
 * whatever it has, and drawn again when this lands. The promise is there for
 * the ones that cannot do that — a batch download, and the warm-up, which
 * walks the grid one frame at a time so that twenty restorations do not all
 * start at once.
 *
 * There is deliberately no "do it anyway" flag. The figures record what they
 * were cut under, and whether they are still current is a comparison against
 * that rather than a judgement each call site has to make for itself — which
 * is what such a flag would be, and it would be wrong at one of them sooner
 * or later.
 */
export function ensureCutout(i) {
    const f = frames()[i];
    if (!f) return Promise.resolve();

    const spec = decorFor(channelFor(i)).cutout;
    if (!spec) {
        // Not this channel's business. Dropped rather than kept in case the
        // user comes back, because the frame they belong to can be reframed,
        // varied and re-uploaded in the meantime and none of that would keep
        // them honest.
        if (f.layers.length) {
            clearCutout(f);
            updatePreview();
            refreshThumb(i);
        }
        return Promise.resolve();
    }

    // What figures of this frame would be cut FROM, right now. The version
    // covers the crop and every photo swap; the preset covers the grade. The
    // mirror is not in it — a flipped figure is the same picture drawn the
    // other way round, done on the canvas and per figure (see toggleFlip).
    // The channel's count, unless the user has taken a figure off — after
    // that, how many this thumbnail shows is their answer and re-cutting must
    // not quietly put it back (see reframe.removeLayer).
    const copies = f.figuresRemoved ? Math.max(1, f.layers.length) : copiesForSlot(spec, i);
    // The cut-out method is in the key because it changes the pixels that come
    // back, exactly as the preset does — see CUTOUT_MODES.
    const key = `${f.version || 0}|${f.editPreset || DEFAULT_PRESET}|${copies}|${cutoutModeFor(i)}`;
    if (f.cutoutKey === key && f.layers.length === copies) return Promise.resolve(f.layers);

    const running = _cuttingOut.get(f.frame_id);
    // A request already out for this frame is only the answer if it is
    // producing the figures wanted NOW. One started under the previous preset
    // is going to come back with the previous preset's pixels and write them
    // in, which would leave the frame looking un-asked — so it is waited for
    // and then asked again, rather than handed back as though it were this
    // request. (Waited for rather than raced: two segmentations of one frame
    // at once is work the second one makes pointless.)
    if (running) {
        return running.wanted === key
            ? running.promise
            : running.promise.then(() => ensureCutout(i));
    }
    const entry = { wanted: key };
    entry.promise = cutOut(i, f, spec, key, copies)
        .finally(() => { if (_cuttingOut.get(f.frame_id) === entry) _cuttingOut.delete(f.frame_id); });
    _cuttingOut.set(f.frame_id, entry);
    return entry.promise;
}


// ── The stage a frame stands on ───────────────────────────────────────────
//
// A channel that owns more than one backdrop lets the user say which of them
// a given thumbnail is shot on, and says it with one button rather than a
// picker: two stages are a thing you flip between while looking at the
// subject standing on them, not a list you go and choose from. The button
// wraps, so the second click on a two-stage channel is the way back.

/** How many backdrops the channel frame `i` is drawn in offers. */
const stageCount = (i) => {
    const bg = decorFor(channelFor(i)).background;
    return (bg && bg.textures) ? bg.textures.length : 0;
};

/**
 * Moves frame `i` onto the next of its channel's backdrops.
 *
 * Per frame, like the gradient and the flips: which stage suits a thumbnail is
 * an answer about the person standing on it, and a subject in a dark jacket
 * that vanishes into one backdrop reads perfectly on the other.
 */
export function cycleBackground() {
    const i = selectedIndex();
    const f = frames()[i];
    const count = stageCount(i);
    if (!f || count < 2) return;
    f.edited = true;
    f.background = (backgroundFor(i) + 1) % count;
    syncBackgroundButton();
    updatePreview();
    refreshThumb(i);
}

/**
 * Shows the button on a channel with a stage to change TO, and hides it
 * everywhere else — on a one-backdrop channel it would be a button whose only
 * possible effect is redrawing the same picture.
 */
export function syncBackgroundButton() {
    const btn = el("bgBtn");
    if (!btn) return;
    btn.classList.toggle("hidden", stageCount(selectedIndex()) < 2);
}

// ── The contour around the figures ────────────────────────────────────────
//
// One button walking a list rather than a colour picker, for the same reason
// the backdrop is one: there are two colours, they are the channel's, and the
// question the user is actually asking is "does this subject need lifting off
// the stage, and warm or cool" — which is a thing to try, not a value to set.
//
// Off is the step PAST the last colour rather than an entry in the list, so a
// channel adds a third colour by writing it in the pack and nothing here
// changes. See `glow` in channels.js.

/** The contour colours the channel frame `i` is drawn in offers. */
const glowColors = (i) => {
    const spec = decorFor(channelFor(i)).cutout;
    return (spec && spec.glow && spec.glow.colors) || [];
};

/**
 * Walks frame `i` to the next contour colour, and off the end of the list back
 * to none.
 *
 * Every figure on the frame at once: they are the same subject cut from two or
 * three moments of the same shot, and a contour is what separates that subject
 * from the stage. Three of them in three colours would be three people.
 */
export function cycleFigureGlow() {
    const i = selectedIndex();
    const f = frames()[i];
    const colors = glowColors(i);
    if (!f || !colors.length) return;
    f.edited = true;
    // ...+ 1 for the step that is "none", which is why this is not just the
    // length of the list.
    f.figureGlow = (figureGlowFor(i) + 1) % (colors.length + 1);
    syncFigureGlowButton();
    updatePreview();
    refreshThumb(i);
}

// ── How the subject is cut out ────────────────────────────────────────────
//
// Two ways of separating a person from a stage, one button, wrapping — the
// same shape as the backdrop and the contour above, and for the same reason:
// which one is right for a given photograph is a thing to TRY while looking at
// the result, not a value anybody can set from a description.
//
// Numbered rather than named because what differs is whether a
// person-segmentation prior gates the saliency map, which is not a sentence
// any user should have to read, and every honest short name for it
// ("tight"/"loose", "person only"/"with props") is wrong for some frame:
// method 2 keeps a ventriloquist's dummy AND the lamp behind it, method 1
// drops both.
//
// There is no button any more, and no cycling: every frame is cut by method 2
// (see `cutoutMode` in state.js, which carries the history). The list stays
// because the request is built from it and because the alternative is the flag
// hard-coded at the call site — which is what it was before, and what made the
// premise behind it impossible to check.

export const CUTOUT_MODES = [
    { request: { keep_props: false } },
    { request: { keep_props: true } },
];

const cutoutMode = (i) => CUTOUT_MODES[cutoutModeFor(i) % CUTOUT_MODES.length];

/**
 * Shows the button on a channel that offers a contour, and names the colour
 * the frame is wearing rather than the one the next click would bring.
 *
 * A button that reads "Gold contour" while the subject is outlined in blue is
 * a button describing its own next press, which nobody reads it as: what a
 * user checks a control for is the state they are in.
 */
export function syncFigureGlowButton() {
    const btn = el("figureGlowBtn");
    if (!btn) return;
    const i = selectedIndex();
    const colors = glowColors(i);
    btn.classList.toggle("hidden", !colors.length);
    if (!colors.length) return;
    const spec = decorFor(channelFor(i)).cutout.glow;
    const at = figureGlowFor(i);
    const on = at < colors.length;
    btn.className = "btn " + (on ? "btn-active" : "");
    // The pack names its colours (see `glow.labels`) because "Contour 1" tells
    // the user nothing and the hex tells them less. A channel that leaves them
    // unnamed falls back to the position, which is at least true.
    const labels = spec.labels || [];
    btn.textContent = on ? `Contour: ${labels[at] || at + 1}` : "Contour: off";
}


async function cutOut(i, f, spec, key, copies) {
    const preset = f.editPreset || DEFAULT_PRESET;
    // Re-asked after every await: an enhance pass, a reframe drop or a preset
    // click landing while these requests are out means what comes back was cut
    // from a picture the frame no longer holds.
    const stale = () =>
        `${f.version || 0}|${f.editPreset || DEFAULT_PRESET}|${copies}|${cutoutModeFor(i)}` !== key;

    const pictures = [];
    try {
        for (let copy = 0; copy < copies; copy++) {
            // One request per figure, and in order rather than all at once.
            // Each one can be a whole restoration of a different photo on the
            // one GPU, so firing three together only makes the backend queue
            // them while holding three responses' worth of memory.
            // A figure the user has walked onward with Variation keeps the
            // moment they left it on; the rest take the channel's own spread.
            //
            // ...unless keeping it would put two figures on the same moment.
            // That is a latch, and it is how a three-figure slot ends up
            // showing one photograph three times and staying there: if the
            // moments could not be found the first time a frame was cut — a
            // search that came back empty, a slot cut while the video was
            // still being scanned — all three figures are handed copy 0, and
            // every re-cut afterwards reads that back and asks for 0 again.
            // The pictures never recover, however many alternates the video
            // turns out to have, because nothing ever asks for a different one.
            //
            // Falling back to the slot's own index breaks it: the spread this
            // channel asks for is copies 0, 1, 2, so an index is always a
            // moment no other figure is holding.
            const held = f.layers[copy] && f.layers[copy].copy;
            const clash = held !== undefined
                && f.layers.some((l, k) => k !== copy && l.copy === held);
            const wanted = (held !== undefined && !clash) ? held : copy;
            const res = await postImage("/cutout-frame", {
                frame_id: f.frame_id, fidelity: FIDELITY, preset, copy: wanted,
                ...cutoutMode(i).request,
            });
            if (!res.ok) {
                // 422 is the honest answer "there is no subject in this frame
                // to separate", which is a fact about the photo and not a
                // fault. Said in the status line rather than left as a frame
                // that silently never gets a person on it.
                if (res.status === 422 && selectedIndex() === i) status(res.detail);
                pictures.forEach(p => URL.revokeObjectURL(p.src));
                return null;
            }
            if (stale()) {
                discardImage(res);
                pictures.forEach(p => URL.revokeObjectURL(p.src));
                return null;
            }
            const img = await loadImage(res.url);
            if (stale()) {
                URL.revokeObjectURL(res.url);
                pictures.forEach(p => URL.revokeObjectURL(p.src));
                return null;
            }
            pictures.push({
                src: res.url, _img: img, natW: img.width, natH: img.height,
                // Which moment of the video this figure came from, and how
                // many others there are — what editor.varyFigure walks.
                copy: (res.meta && res.meta.copy) || 0,
                moments: (res.meta && res.meta.moments) || 0,
            });
        }

        // The placements survive a re-cut whenever there are the same number
        // of figures to place: switching the edit preset must not throw away
        // an arrangement the user spent time on, and between two presets each
        // figure is the same person at very nearly the same size — the trim
        // moved by two pixels, measured. Only the PICTURES are replaced.
        const previous = f.layers;
        if (previous.length === pictures.length) {
            f.layers = previous.map((layer, k) => ({
                ...layer, ...pictures[k],
                _alpha: null,   // measured from the picture, and the picture just changed
            }));
        } else {
            f.layers = fitCutout(spec, pictures);
            f.activeLayer = -1;
        }
        previous.forEach(layer => { if (layer.src) URL.revokeObjectURL(layer.src); });
        f.cutoutKey = key;

        if (selectedIndex() === i) {
            updatePreview();
            // The zoom control now means this figure's size, and the slider's
            // bounds are that figure's own — see reframe.cutoutZoomOf.
            syncZoomSliderFor(f.frame_id);
        }
        refreshThumb(i);
        return f.layers;
    } catch (_) {
        // network / server error — the frame keeps whatever it had, and the
        // next selection or edit asks again
        pictures.forEach(p => URL.revokeObjectURL(p.src));
        return null;
    }
}

/**
 * Cuts every frame in the grid out, one at a time, in the background.
 *
 * The grid is twenty options and this channel's options are its arrangements —
 * which of them show the subject once, twice or three times, and how each of
 * those looks. A card that shows an empty stage until it is clicked is not an
 * option anybody can choose between, so the whole grid is filled in as soon as
 * it exists rather than on demand.
 *
 * Strictly one at a time. Each frame is a restoration of its photo (measured
 * at ~3 seconds cold, and a two- or three-figure slot pays that once per
 * figure), the backend serialises them into the model anyway, and firing
 * twenty at once would only mean the user's own click waits behind nineteen
 * others instead of behind one. The results are cached backend-side, so the
 * work is paid for once and every later click on those frames is instant (see
 * session.cutouts).
 *
 * Abandoned the moment a new video replaces the grid: `frames()` returns a
 * different array then, and the frames this was walking no longer exist.
 */
export async function warmCutouts() {
    const grid = frames();
    for (let i = 0; i < grid.length; i++) {
        if (frames() !== grid) return;
        await ensureCutout(i);
    }
}

/**
 * Runs the edit over every frame in the grid, ahead of the user asking for any
 * of them.
 *
 * The counterpart of warmCutouts, and the same shape for the same reasons —
 * strictly one at a time, and abandoned the moment a new video replaces the
 * grid. What differs is that this one is not a warm-up. On a frame fetch the
 * edit is the product: the expected path through the mode is two clicks,
 * process and then download all, with the user never opening a single frame.
 * If the pass only ran on selection, that second click would either hand back
 * unedited frames or begin the entire job from cold.
 *
 * So it is reported rather than done quietly. This is minutes of work on a
 * grid of forty, and a status line counting them off is the difference between
 * a machine that is busy and one that looks broken.
 */
export async function applyEditToAll() {
    const grid = frames();
    for (let i = 0; i < grid.length; i++) {
        if (frames() !== grid) return;
        status(`Applying the edit — frame ${i + 1} of ${grid.length}`);
        await ensureEnhanced(i);
    }
    if (frames() !== grid) return;
    status(`${grid.length} frames ready. Download all, or pick the ones you want.`);
}

// ── Image overlay ─────────────────────────────────────────────────────────

/**
 * Redraws the side panel's list of the frame's image overlays.
 *
 * One chip per overlay, newest at the top of the stack shown last, each
 * carrying its own delete. Uploading stays available whatever is already
 * there — adding an overlay is the point, and a button that turned into
 * "Replace" was the thing making a second one impossible.
 *
 * Clicking a chip's picture makes that overlay the active one, which is the
 * only way to reach an overlay lying completely underneath another: on the
 * canvas the topmost one takes every press.
 */
export function syncLogoControls() {
    const list = el("logoList");
    if (!list) return;
    const idx = selectedIndex();
    const logos = logosFor(idx);
    const f = current();

    list.innerHTML = "";
    for (let i = 0; i < logos.length; i++) {
        const logo = logos[i];
        const chip = document.createElement("div");
        chip.className = "logo-chip" + (f && f.activeLogo === i ? " active" : "");
        chip.innerHTML = `<img src="${logo.src}" alt="">` +
                         `<button class="logo-chip-del" title="Delete this image">&times;</button>`;
        chip.querySelector("img").onclick = () => selectLogo(i);
        chip.querySelector(".logo-chip-del").onclick = (e) => {
            e.stopPropagation();
            removeLogoAt(i);
        };
        list.appendChild(chip);
    }
    el("logoListHint").style.display = logos.length ? "" : "none";
}

/** Moves the on-canvas handles onto one of the frame's overlays. */
export function selectLogo(i) {
    const f = current();
    if (!f || !f.logos[i]) return;
    f.activeLogo = i;
    f.titleActive = false;   // one thing wears the handles — see reframe.selectTitle
    f.activeLayer = -1;
    syncLogoControls();
    updatePreview();
}

export function removeLogoAt(i) {
    const f = current();
    if (!f || !f.logos || !f.logos[i]) return;
    f.edited = true;
    f.logos.splice(i, 1);
    // The handles have to land somewhere real: on whatever slid down into
    // this slot, or on the new top of the stack when the last one went.
    f.activeLogo = Math.min(f.activeLogo, f.logos.length - 1);
    syncLogoControls();
    updatePreview();
    refreshThumb(selectedIndex());
}

export async function handleLogoFile(file) {
    const f = current();
    if (!f) return;
    const dataUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error("could not read file"));
        reader.readAsDataURL(file);
    });
    const img = await loadImage(dataUrl);
    const targetW = CW * 0.28;                        // default on-canvas width
    const scale = Math.min(1, targetW / img.width);   // never upscale past the upload's own resolution
    f.edited = true;
    // Appended, not assigned: a frame carries as many overlays as the user
    // adds, and the newest goes on top of the stack.
    f.logos.push({
        src: dataUrl, _img: img,
        natW: img.width, natH: img.height,
        x: CW / 2, y: CH / 2,   // centered, per spec
        scale,
    });
    f.activeLogo = f.logos.length - 1;   // the one just added is the one you want to place
    syncLogoControls();
    updatePreview();
    refreshThumb(selectedIndex());
}

// ── Batch ticking ─────────────────────────────────────────────────────────

export function togglePick(i, onCarryOver) {
    const picking = !frames()[i].picked;
    // Batch-ticking a frame is also its first real "touch" if it's never been
    // clicked into — without this, a frame only ever ticked (never selected)
    // stayed on blank defaults forever, so its download came out without the
    // title/logo/preset the user set up on the frame they were actually
    // looking at. Selection's own carry-over doesn't cover this path since
    // ticking deliberately avoids switching the preview.
    if (picking && onCarryOver) onCarryOver(i);
    frames()[i].picked = picking;
    el(`pick-${i}`).classList.toggle("on", picking);
    if (picking) refreshThumb(i);   // reflect the inherited title/logo on the grid card too
    syncBatchButton();
}
