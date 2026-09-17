/**
 * The channels this tool makes thumbnails for, and the brand guidelines each
 * one is set with.
 *
 * Everything the title overlay used to hardcode — the face, the white on
 * black-shadow lines, the red highlight box, the 4% left margin, the 50%
 * target width — was in fact one channel's house style written into the
 * renderer. It moved out of text.js into a list here, and it has now moved
 * one step further, out of this file and into `content/channels/`.
 *
 * Which leaves this module as two things and no data:
 *
 *   - the shared skeleton every channel starts from, which IS code: it is
 *     the set of knobs the renderer knows how to read, and adding one means
 *     teaching text.js what to do with it;
 *   - the loader that reads the packs, merges them onto that skeleton, and
 *     hands the rest of the app the same synchronous API it always had.
 *
 * The point of the split is the install. A channel is now a folder of JSON
 * that a patch drops in — or that a user drops in by hand — with no rebuild,
 * no reinstall, and no edit to any file that ships as code. See
 * content/README.md.
 *
 * Nothing is applied until a channel is picked: with none selected the title
 * fields are inert and no text is drawn (see editor.js's channel lock and
 * addTextOverlay's guard). A thumbnail always belongs to some brand, and
 * "whatever the last channel's font happened to be" is not one.
 *
 * The selection is per-FRAME. It used to be per-page, on the reasoning that a
 * batch is made for one channel at a time and letting frame 3 carry a
 * different brand from frame 4 would only ever be a mistake the UI made easy.
 * That was wrong about what a batch is: the frames of one video are worked
 * through one at a time, and trying a different channel on a later frame
 * reached back and re-branded every frame already finished. A choice made
 * while looking at one frame has to be a choice about that frame.
 *
 * What lives here is still a single "active" channel, because everything that
 * draws a title reads it at draw time and threading a style through every
 * measurement in text.js would be a far larger change than this is worth. It
 * now means "the channel being drawn with right now", which is the selected
 * frame's while the user is working, and each frame's in turn while the grid
 * of thumbnails is rebuilt (see usingChannel, and state.channelFor).
 */

// Where the frontend server publishes the merged content document — the
// shipped packs under app/frontend/content/ and the machine's own under
// <install>/content/, in one fetch. See serve.py.
const CONTENT_URL = "content/channels.json";

// ── Faces ─────────────────────────────────────────────────────────────────
// `family` is the name the canvas asks for and fontguard verifies actually
// arrived; `file` is where it is served from, named here only so a
// missing-font message can say which file to look for. `stack` is what Fabric
// is handed — the fallbacks in it are a damage-limitation path, never a
// design choice: a title laid out against Impact's metrics is broken, not
// merely different (see fontguard.js).
//
// A face declared by a pack is registered with the browser from its JSON at
// load time (registerFace below), which is what lets a channel bring its own
// font without an @font-face being added to styles.css by hand.

/**
 * The face used for measurements taken before any channel exists — the one
 * the app ships with and the one styles.css preloads.
 *
 * Hardcoded rather than read from the packs because it has to be available
 * synchronously, at module scope, to code that runs before the fetch below
 * has resolved. The pack of the same id overrides it the moment content
 * loads; if content never loads, this is what the app degrades to instead of
 * having no face at all.
 */
const BUILTIN_FACE = {
    id: "trade-gothic-heavy-compressed",
    family: "Trade Gothic Next LT Pro Heavy Compressed",
    file: "fonts/TradeGothicNextLTProHeavyCompressed.otf",
    weight: "900",
    stack: "'Trade Gothic Next LT Pro Heavy Compressed', Impact, 'Arial Black', sans-serif",
    css: true,
};

// ── Shared skeleton ───────────────────────────────────────────────────────
/**
 * Structural defaults every channel starts from, so a channel definition
 * states only what makes it that channel.
 *
 * The layout numbers are ratios of the 1280×720 canvas rather than pixels —
 * the same reason every other measurement in the app is (see config.CW/CH).
 * `padRatio` is a fraction of the line's own font size, not of the canvas:
 * the highlight box has to keep its proportions whatever size the line came
 * out at.
 */
const BASE_TITLE_STYLE = {
    face: BUILTIN_FACE,
    lineHeight: 1,
    uppercase: true,
    color: "#FFFFFF",
    // Drop shadow behind unhighlighted lines — what keeps white type legible
    // over a bright photo.
    shadow: { color: "rgba(0,0,0,0.8)", blur: 12, offsetX: 3, offsetY: 3 },
    layout: {
        leftRatio: 0.04,            // margin from the canvas edge the block starts at
        widthRatio: 0.50,           // width the LONGEST line is set to
        maxBlockHeightRatio: 0.80,  // block is pulled in if it would exceed this
        lineGap: 10,                // gap between the visible edges of consecutive lines
        // The size every line is set at, in canvas pixels — or `null`, which
        // is every channel that makes a thumbnail, meaning the size is fitted
        // to the block instead.
        //
        // Fitting is the right answer for a headline. A headline is three or
        // four words that were WRITTEN to be a headline, the block is one of
        // the two things in the frame, and the type should be as large as the
        // frame will let it be — so the renderer solves for the size and the
        // brand states the box.
        //
        // It is the wrong answer for a caption. A caption is whatever sentence
        // happened to be spoken over the still, its length is not a decision
        // anybody made, and a fitted size turns that accident into a visible
        // one: a short sentence comes back set enormous and a long one small,
        // so a grid of forty stills cut from one video is forty different type
        // sizes. Burned-in speech does not work that way anywhere it is
        // actually used — it is one size, always, and the sentence takes
        // however many lines it takes.
        //
        // Stating a number here says the size is the brand's and not the
        // sentence's. Three things follow from it, and all three are the point
        // rather than side effects:
        //
        //   - `uniformSize` stops meaning anything. There is no per-line fit
        //     left to make uniform; every line is at this size already.
        //   - the height budget stops pulling the block in. `maxBlockHeightRatio`
        //     shrinks a block that overflows by re-fitting it smaller, and a
        //     fixed size the renderer is free to shrink is not a fixed size. A
        //     long caption makes a taller block instead, and the user drags it
        //     where they want it.
        //   - the line breaks stop being chosen by height. caption.js breaks a
        //     sentence over the MOST lines that fit, because more lines meant
        //     bigger type; at one fixed size it breaks over the fewest lines
        //     that fit the width, which is ordinary wrapping (js/caption.js).
        fontSize: null,
        // The widest a line may be in CHARACTERS, or `null` for a channel
        // that measures its lines and nothing else.
        //
        // A second ceiling on the line breaks, next to the block's own width,
        // and the two are not the same question. Width asks what fits; this
        // asks what reads. A caption set at a stated size can hold a great
        // many characters on one line and still fit — 40px of Arial Black
        // across 486px is nearly twenty of them, and a face half as wide
        // would be forty — and forty characters is not a caption line any
        // more, it is a paragraph rule. Somebody watching this on a phone
        // reads a line at a glance or does not read it.
        //
        // Characters and not words, because it is a length the eye measures
        // and not a grammar: three long words and six short ones are the same
        // amount of reading.
        //
        // Only caption.js reads it (the renderer never re-breaks a line, and
        // the user's own breaks are theirs whatever this says), so it does
        // nothing at all on a channel whose titles are typed rather than
        // transcribed.
        maxChars: null,
        // How the block resolves the tension between two things it cannot
        // have at once, since a line's width is set by its character count:
        //
        //   true  — one font size for every line. Every highlight box comes
        //           out the same height whatever gets typed, and a short line
        //           is simply short — centred in the block, so it gives up
        //           half its slack at each end (see inkLeft in text.js).
        //   false — each line is fitted to the full width at its own size, so
        //           every line spans the same width and both edges are flush.
        //           Box height then follows each line's own size, so a short
        //           line's box is taller than a long one's.
        //
        // Neither is a default worth calling correct — it is a brand's
        // decision about which edge of its titles is the straight one, which
        // is why it sits in the pack rather than in the renderer.
        uniformSize: true,
        // Where the line breaks come from.
        //
        //   false — three fixed slots, one field each. The block is whichever
        //           of them have something in them, and an empty slot is not
        //           a line at all.
        //   true  — one block of text, and the breaks in it are the user's.
        //           Every break is reproduced exactly as typed, including a
        //           blank line, which comes out as a blank line rather than
        //           being closed up (see spacer rows in text.js).
        //
        // It is a decision about the titles a channel writes, not about the
        // UI: three slots suit a headline built as three beats, and a channel
        // whose titles are sentences cannot say where they break inside them.
        // The panel follows this — see editor.syncTitlePanelMode.
        freeform: false,
        // How many of those slots there are. Three for every channel that has
        // ever existed here, which is why it was a fact about the markup
        // rather than a number until a channel wanted four.
        //
        // Capped by how many fields the page actually has (config.LINE_IDS) —
        // a pack asking for more gets what there is, rather than a title whose
        // last line can be stored but never typed. Ignored entirely by a
        // freeform channel, where the breaks are the user's and there are no
        // slots to count.
        lines: 3,
        // Where a line NARROWER than its neighbours takes its slack, inside
        // the block: "center", "left" or "right".
        //
        // Centred by default, because with the slack split evenly neither edge
        // claims to be the straight one — flush to one side, a short line
        // reads as a line that has slipped rather than as a shorter line. A
        // channel overrides it when one of its edges is a real edge on screen
        // and not one the eye has to infer: see the note in text.js, and the
        // Laugh Society pack, whose block is set hard against the right margin
        // because the left of its canvas belongs to the subject.
        //
        // Not the same question as `uniformSize`. That one decides whether
        // there IS any slack — fitted individually, every line reaches the
        // full width and this changes nothing; at one shared size the short
        // lines stay short and this is what says where they sit.
        align: "center",
        // Where the block sits DOWN the canvas: "center", "bottom" or "top".
        //
        // Centred everywhere until a channel wanted otherwise, and centred is
        // the right default for a title that is the thumbnail's headline —
        // the photograph is composed around it and the block is one of the
        // two things in the frame. A caption is not that. It is words laid
        // over a picture that was composed without it, it belongs where the
        // eye expects burned-in speech to be, and that is an END of the frame
        // whatever the picture happens to contain.
        //
        // Which end is the channel's to say, and "top" is the mirror of
        // "bottom" rather than a third idea: a vertical still has a subject
        // across the middle of it, so the two places a caption can go without
        // covering a face are the head of the frame and the foot of it. The
        // foot is where a player's own captions appear and is what every pack
        // under the Snapchat heading wanted until one was cut for the other
        // end (see conspiracy-central).
        //
        // This is the block's STARTING place, not a rule about where it stays:
        // a `movable` channel stores the user's drag as an offset from
        // whatever this puts it at (see frame.titleBox).
        vAlign: "center",
        // How much room is left under the block when vAlign is "bottom", as a
        // fraction of canvas height. Ignored when it is centred.
        //
        // A margin rather than a position, because the block grows upward from
        // it: a caption that gains a fourth line has to stay the same distance
        // off the bottom edge, or the type would walk down the frame as the
        // sentence got longer.
        bottomRatio: 0.06,
        // The same thing at the other edge, read when vAlign is "top" and
        // ignored otherwise exactly as `bottomRatio` is.
        //
        // Also a margin, and this one needs no argument for being one: a block
        // hung from the top edge grows DOWNWARD, so its first line stays where
        // it is however long the sentence turns out to be and there is nothing
        // to subtract to keep the distance constant. That is why the two are
        // two keys rather than one signed number — the arithmetic is not the
        // same at both ends (see blockTop0 in text.js), and a channel states
        // whichever edge its design is measured from.
        //
        // Measured to the first thing DRAWN, not to the letters: on a channel
        // whose highlight is a box, a first line with a picked word in it puts
        // the box's top edge on this margin and the capitals a padding's worth
        // below it. That is the distinction `bottomRatio` already makes at the
        // other edge under `panel.scope: "line"`, and it is the same size —
        // the padding, and nothing else.
        topRatio: 0.06,
    },
    // A drop shadow behind the type. `null` for a channel whose titles sit on
    // something that already separates them from the photo — see `panel`.
    // `null` for a channel that has no highlight feature at all: the line
    // buttons go dead (see editor.renderHlButtons) and every line is set the
    // same way, which is a different thing from a highlight that happens to
    // look like the rest of the block. A brand whose lines are all one
    // treatment says so here rather than being given a highlight nobody can
    // tell apart from the type.
    highlight: {
        // What highlighting a line DOES:
        //   "box"   — a filled rectangle behind the line, the line set in
        //             `color` on top of `fill`.
        //   "color" — no rectangle at all; the line is simply set in `color`
        //             instead of the block's usual colour. `fill` and
        //             `padRatio` go unused.
        //   "texture" — as "color", but the letters are filled with the image
        //             at `texture` instead of a flat colour. The image is
        //             mapped across the WHOLE block, not per line, so several
        //             highlighted lines read as one sheet of it rather than
        //             as several copies. `color` is what gets drawn until the
        //             image has loaded, and if it never does.
        //   "gradient" — as "texture", but the surface is a ramp stated in
        //             `gradient` rather than a picture. Mapped across the
        //             whole block for the same reason and to the same effect:
        //             two highlighted lines are two windows onto one ramp, not
        //             two copies of it stacked. `color` is the flat fallback
        //             for a pack that names the mode and forgets the ramp.
        mode: "box",
        // What one highlight COVERS:
        //   "line" — the whole line, as one box or one recolour.
        //   "word" — one word, and only the words the user picked. Each gets
        //            its own box, or its own recolour, sized to that word's
        //            own ink. Words picked side by side are one run and get
        //            one box between them, because that is what a highlighter
        //            does; words either side of a line break are not.
        //
        // Works in every mode. It used to be "box" only, on the reasoning that
        // a per-word recolour was a different feature nobody had asked for —
        // and then a channel did (see hall-of-femme, whose card is already a
        // solid ground and has nothing to box against). What a colour-scope
        // word highlight does is exactly what the boxed one does minus the
        // rectangle: the picked run is set in `color`, its neighbours in the
        // block's.
        //
        // The cost of picking words rather than lines is that a line can no
        // longer be handed to the renderer as one string, which is what the
        // spacing between two words comes from. Two ways out, and which one is
        // taken depends on whether `face` below is set:
        //
        //   no face — the line IS still drawn as one string, and the picked
        //             runs are drawn a second time over their own boxes. The
        //             spacing is the font's, untouched, because nothing was
        //             ever split.
        //   a face  — the line is drawn as a run per treatment, each in its
        //             own face, each placed at the pen position the runs
        //             before it advanced to. The spacing WITHIN a run is the
        //             font's; the one place arithmetic decides anything is the
        //             space between two runs, which is a space between two
        //             faces and has no kerning pair to lose.
        scope: "line",
        fill: "#FF070C",
        color: "#FFFFFF",
        // A face for the picked lines alone, named or written inline exactly
        // as `title.face` is. `null` — every channel but one — means the
        // highlight is set in the block's own face and differs from the rest
        // of the title only in colour.
        //
        // It is a second face on one block of type, which is normally the
        // thing a house style exists to prevent. It earns its place where the
        // highlight is not a colour laid over the same voice but a different
        // voice: Laugh Society 2 sets its titles in a slanted face and picks
        // its lines out in an upright one, so the picked line reads as spoken
        // by someone else rather than merely lit differently.
        //
        // Everything measured for a picked line is measured in THIS face —
        // the width fit, the ink bounds, the cap band the box is padded from
        // — because two faces on one block means two sets of metrics and a
        // line laid out against the wrong one is the failure fontguard exists
        // to catch, only self-inflicted. See `styleFor` in text.js.
        //
        // Works under either scope, and the renderer changes technique
        // between them (see `scope` above). Under line scope the row is one
        // string in one face. Under word scope it becomes a run per
        // treatment, each measured, stacked and drawn in its own face — which
        // is what a face that changes halfway along a line costs, and it is
        // paid only by the channels that ask for one.
        face: null,
        // Only read in "texture" mode: the image the letters are filled with.
        texture: null,
        // Only read in "gradient" mode: the ramp the letters are filled with,
        // mapped across the whole block.
        //
        //   type    "linear" or "radial". A radial ramp is a ROUND one — the
        //           colour spreads out from a point in rings, so what a line
        //           gets depends on how far it sits from that point in any
        //           direction rather than on how far up the block it is. It is
        //           the difference between type that fades and type that is
        //           lit.
        //   from    where it starts, as fractions of the block's own box, so
        //           [0.5, 1] is the middle of its bottom edge. In "linear" it
        //           is one end of the line; in "radial" it is the centre the
        //           rings come out of.
        //   to      the other end, in "linear" only.
        //   radius  how far out the last stop lands, in "radial" only, as a
        //           fraction of the block's LONGER side — so a ramp reaching
        //           the far corner is a little over 1 and not a number that
        //           has to be recomputed every time a title gains a line.
        //   stops   [{at, color}, …], `at` running 0 to 1 from `from`.
        gradient: null,
        padRatio: 0.12,
        // How far the box's corners are rounded, in canvas pixels. Zero — a
        // hard rectangle — everywhere it is not stated.
        //
        // Pixels and not a ratio, unlike `padRatio` beside it, and the
        // difference is deliberate. Padding is part of the type: it is the air
        // around the letters, so it has to grow with them or the box's
        // proportions change with every line. A corner radius is part of the
        // GRAPHIC. It is the same 13px whether the caption came out four words
        // long or fourteen, for the same reason the rounding on a button does
        // not depend on what the button says — and a radius that scaled with
        // the type would give one caption soft corners and the next one sharp.
        //
        // Clamped to half the box at the draw, so a radius larger than the box
        // it is rounding comes out as a capsule rather than as nothing.
        radius: 0,
        shadow: { color: "rgba(0,0,0,0.7)", blur: 20, offsetX: 0, offsetY: 0 },
    },
    // A halo behind every letter: the glyphs redrawn without their outline and
    // blurred, under the real ones. `null` for a channel that doesn't glow.
    // Unlike `shadow`, this takes no offset — it is light coming off the
    // letters, not a shadow they cast, so it sits centred on them.
    //
    // It is drawn, never measured: like a shadow, it spreads past the block
    // without being part of it, so no line position or width depends on it.
    //
    // `toggle: true` inside it makes the halo an OFFER rather than part of
    // the channel: the parameters are the channel's, but whether it is drawn
    // is the user's, per frame, from a button in the panel (see
    // editor.toggleGlow and frame.glow). Everything else about it is
    // unchanged — a channel that does not say `toggle` glows always, which is
    // what a channel whose look includes the halo means.
    glow: null,

    // Whether the user may move and resize the title on the canvas.
    //
    // Off everywhere by default, and that is not timidity: the layout numbers
    // in `layout` ARE the brand's guidelines, and a title that can be dragged
    // anywhere is a title that no longer has any. A channel turns it on when
    // its titles are composed against whatever is in the photo rather than
    // placed to a rule — and even then the guidelines are where the block
    // starts, with the drag stored as an offset from them (see frame.titleBox
    // and the `placement` note in text.addTextOverlay).
    movable: false,

    // An outline around every letter, highlighted or not. `width` is how much
    // of it you SEE — the renderer doubles it, because a canvas stroke
    // straddles the letterform and the inner half is then painted over by the
    // fill. `opacity` is the outline's alone; it does not touch the letters.
    //
    // Layout follows from it automatically: an outline is ink, so it widens
    // and heightens the line, and every measurement in text.js is taken
    // through the same draw path that will produce it (see textProps).
    outline: null,

    // A SECOND outline, far wider than the first and drawn under both it and
    // the letters — the sticker cut a title is set on. At this width the
    // stroke around one letter runs into the stroke around its neighbours,
    // and around the lines above and below, so what reaches the eye is not an
    // outline at all but one solid shape with the type sitting in it.
    //
    // That merging is the whole effect, and it is why this is a second stroke
    // rather than a wider first one: an outline this thick painted in one
    // pass would swallow the keyline that separates the white letters from
    // the slab. Two strokes, the narrow one over the wide one, keep both.
    //
    // Any gap the strokes enclose but do not cover — two words landing just
    // too far apart to touch — is found and filled automatically, so a
    // channel does not have to say anything about it. See holePatch.
    //
    // `fill` is a colour, or a gradient mapped across the WHOLE slab rather
    // than per line — see stickerFill in text.js. Per line, each line would
    // carry its own copy of the ramp and the block would band; one map makes
    // the merged shape read as a single object, which is what it looks like.
    //
    // Drawn, never measured, for the reason `glow` is: the slab spreads past
    // the block in every direction and the lines are deliberately left to
    // overlap inside it, so letting it into the layout would push them apart
    // and take the solid shape back apart into stripes.
    sticker: null,

    // A slab drawn behind the WHOLE block — one rectangle around all the
    // lines together, as opposed to `highlight`, which is per line. `null`
    // for a channel that sets its type straight onto the photo.
    //
    // Drawn as two rectangles rather than one stroked one: a canvas stroke
    // straddles the path, half of it inside, so "a 6px border" would really
    // be 3px of border eating 3px of fill. A `border`-sized rectangle with
    // the fill inset by that much makes the number mean what it says.
    //
    // A pack states the whole slab or none of it — `fill`, `border.width`,
    // `border.color`, `padding`, `shadow` — because the skeleton's answer here
    // is `null` and there is nothing underneath for a half-written panel to
    // fall back onto. Two keys are optional:
    //
    //   radius  absent, the slab has square corners; a number rounds them by
    //           that many canvas pixels. The inner rectangle takes the
    //           border's width off it, so a rounded slab with a border comes
    //           out as two corners sharing a centre rather than as a rounded
    //           fill sitting in a square keyline.
    //   scope   "block" (absent, and what a panel has always been) draws ONE
    //           rectangle around the whole title. "line" draws one per line,
    //           each the width of that line — the block stops being a card
    //           with type on it and becomes a shape that follows the
    //           sentence, which is what burned-in social captions look like:
    //           a two-word line reads as a short line instead of as a long
    //           line with air on both sides.
    //   joined  read under "line" only. False (absent) leaves those boxes
    //           SEPARATE — a stack of tags with `lineGap` of photograph
    //           showing between them. True overlaps them into ONE continuous
    //           background: still stepping in and out with the sentence, but
    //           with a single outline and a single shadow round the whole of
    //           it rather than one per line.
    //
    // `scope: "line"` changes three things besides the drawing, and all three
    // are what makes it a look rather than a decoration:
    //
    //   - the boxes are what the rows are STACKED by, so `bottomRatio` is the
    //     margin under the last one rather than under the last line. What
    //     `lineGap` measures depends on `joined`: separate, it is the gap
    //     between two BOXES, because that is what is visible between two
    //     lines; joined, there is no such gap by construction, so it is the
    //     gap between the lines themselves. A joined panel therefore wants a
    //     far smaller number than a separated one — it is spacing letters,
    //     not slabs — and a pack that turns `joined` on without dropping
    //     `lineGap` gets a block with holes of photograph punched through it.
    //   - every box takes one shared vertical band, so they differ in width
    //     and in nothing else. Sized to their own rows they would come out at
    //     four heights over an accent and a descender, which reads as four
    //     mistakes rather than as one stack.
    //   - a box is sized to the whitespace the user typed as well as to the
    //     ink. A trailing space is a typing accident in a block of type on a
    //     photograph and is ignored everywhere else in this renderer; behind a
    //     box it is the one way of saying "leave room here", which is how a
    //     caption gets a gap for an emoji to sit in.
    panel: null,

    // The colours this channel lets the USER choose between, as opposed to
    // the ones it decides — `color`, `highlight.color` and `panel.fill`
    // above. `null` everywhere by default, and that default is what every
    // channel here did until one wanted otherwise: a brand's colours are its
    // colours, and a picker in front of them is a picker in front of the
    // guidelines.
    //
    //   text       what the letters may be set in. Three lists rather than
    //   highlight  one, what the words picked out of them may be set in, and
    //   panel      what the slab behind them may be filled with — all
    //              independent: the point of offering more than one is the
    //              pairing, so a pack stating the same three colours twice is
    //              stating that any of the nine combinations is allowed
    //              rather than repeating itself. A pack may state any of them
    //              and not the others.
    //
    // `text` recolours the highlight as well as the block UNLESS `highlight`
    // is stated, and that is a decision rather than an omission. A channel
    // offering the choice at all is usually one whose type is a single colour
    // the user picks — where the highlight is a different CUT of the face
    // (see `highlight.face`) and not a different colour, leaving it at the
    // pack's own value would mean choosing pink and getting one line still in
    // white.
    //
    // Stating `highlight` says the opposite in as many words: this channel's
    // picked words ARE a different colour, and which one is the user's too. A
    // pack usually offers the same list twice over, because what it is really
    // offering is the second choice being a different answer to the first —
    // the highlight is whichever of the channel's colours the block is not.
    // The row is hidden on a channel with no highlight at all (see
    // editor.paletteOf), the same way the slab's row is hidden with no slab.
    //
    // Which entry a frame is wearing is the FRAME's (see state.textColorFor,
    // highlightColorFor and panelColorFor), for the reason the gradient and
    // the backdrop are: it is an answer about how one picture reads, and the
    // frames of one run are not all the same picture. All three start at
    // null, which means the channel decides — the pack's own `color`,
    // `highlight.color` and `panel.fill` are its first answer and there is no
    // reason to keep a second copy of that in a list index, free to disagree
    // with it. So a palette does not have to open with the colour the channel
    // is set in, and the swatch that shows as chosen on an untouched frame is
    // whichever one matches it. An index that lands outside the list falls
    // back the same way, which is what a frame carried over from a channel
    // with a longer palette gets.
    //
    // The buttons are the colours themselves — see editor.renderColorButtons.
    // A swatch says what it does and a label naming it does not, which is the
    // opposite of the contour's problem (`cutout.glow.labels`), where the
    // steps are lights on a person and a hex would tell nobody anything.
    colors: null,
};

/**
 * What a channel puts on the canvas besides the title — a backdrop of its own,
 * a brand mark, and the subject cut out of the photograph.
 *
 * All three are null everywhere by default, and that default is the whole of
 * what every channel here did until now: the thumbnail is the photograph, with
 * a title over it. A channel that fills any of these in is saying the
 * photograph is no longer the picture — see the Laugh Society 1 pack, where it
 * is a stage backdrop with the person standing on it and the frame is only
 * where that person is cut from.
 *
 * They live outside `title` because none of them is type, and outside
 * `branding` because that is a set of per-frame defaults the user overrides
 * with a button; these are the channel's furniture and are drawn the same way
 * on every frame.
 */
const BASE_DECOR = {
    // How the frame's OWN photograph is presented, for a channel that does not
    // want it edge to edge.
    //
    // Null everywhere else, and null is not a placement decision — it is the
    // photograph BEING the picture: drawn at 0,0 to the full canvas, which is
    // what every thumbnail in this app has always been and what the crop the
    // backend chose was composed for. A pack that fills this in is saying the
    // frame is a card lying on something rather than the surface itself, and
    // it then has to say what that something is, because the margin it just
    // created is canvas nobody has painted.
    //
    //   fillRatio  how big the card is, as a fraction of the canvas, applied
    //              to BOTH dimensions. One number rather than two because the
    //              photograph arrives already cropped to the canvas's shape
    //              (see capture.output), so scaling both by the same fraction
    //              is the one move that leaves the picture the picture — a
    //              width and a height stated apart would re-crop it, or worse,
    //              squash it, to a shape nobody chose. Centred, for the same
    //              reason: the card is the whole of what a viewer looks at,
    //              and a margin that is not equal on both sides is a
    //              composition, which is a thing a pack should have to say out
    //              loud rather than get by arithmetic.
    //   radius     the corner rounding, in canvas pixels, clamped to half the
    //              card. Pixels rather than a fraction, for exactly the reason
    //              `title.panel.radius` is: the rounding is part of the
    //              graphic and not part of the picture, so it is the same 13px
    //              whatever the card is sized to.
    //   fill       a solid colour painted on the card INSTEAD of the
    //              photograph, for a channel whose card is a ground rather
    //              than a picture. One key rather than a colour plus a switch,
    //              because there is only one thing it can mean: the card is
    //              this colour, so the frame's own photo is not drawn in it.
    //              A channel that states this is saying the photograph reaches
    //              the thumbnail some other way — through the backdrop below,
    //              through the subject cut out of it, or both — and if it
    //              states neither then it has asked for a blank card, which is
    //              a composition and not a fault.
    //              Absent everywhere else, and absent means the card IS the
    //              photograph, which is what a card was for until now.
    //   backdrop   what is painted over the whole canvas UNDER the card, which
    //              is what the margin is. Two kinds, told apart by `source`.
    //
    //              A GRADIENT, which is the form with no `source`: written
    //              {from, to, stops}, where `from` and `to` are corners of the
    //              canvas as fractions of it — [0,0] to [0,1] is top to
    //              bottom — and the stops are the same {at, color} pairs a
    //              scrim's are. Colours carry their own alpha, so a backdrop
    //              that is meant to be partly the photograph says so in rgba()
    //              rather than in an opacity of its own.
    //
    //              Or the FRAME'S OWN PHOTOGRAPH, blurred: {source: "frame",
    //              cropRatio, blur}. `cropRatio` is the share of the still
    //              taken from its centre before it is scaled back out to cover
    //              the canvas — 0.5 is the middle half of it at twice the
    //              size — and `blur` is the radius in canvas pixels. It is
    //              the same picture the card is showing, pushed out of focus
    //              and out to the edges, which is how a vertical still fills a
    //              frame it does not fit without a colour being invented for
    //              the margin. Cropped IN before it is blurred rather than
    //              simply scaled up, because a blur spreads whatever detail is
    //              there over its own radius and a tighter crop is fewer,
    //              larger shapes — the difference between a soft version of
    //              the picture and a grey one.
    //
    //              Stated as a kind rather than as two keys, because it is one
    //              question — what is under the card — with two answers, and
    //              a pack that could give both would be painting one over the
    //              other in an order nobody wrote down.
    //   tint       a second gradient, in the same form, painted OVER the card
    //              and clipped to it — the same rounded rectangle, so it stops
    //              exactly where the picture does. That is the whole of the
    //              difference between this and a scrim: a scrim lies across
    //              the canvas and would run out over the backdrop, where this
    //              is a treatment OF the photograph and has to end with it.
    //              Measured in the card's own box, so [0,1] to [0,0] runs from
    //              the bottom edge of the picture to its top edge and not from
    //              the bottom of the frame.
    //
    // Drawn under the figures, the brand marks and the title, all of which
    // still work in canvas coordinates: the card is where the PHOTOGRAPH goes,
    // not a frame everything else is composed inside. A title is therefore
    // free to stand on the backdrop below the card, which is what a channel
    // whose caption sits at the foot of the frame gets, and it is a placement
    // rather than an accident — the margin is part of the design or the pack
    // should not have asked for one.
    photo: null,

    // Drawn under everything, INSTEAD of the frame's photo. `texture` is a
    // path resolved like any other pack asset. Scaled to cover the canvas and
    // centre-cropped, never stretched — a backdrop distorted to 16:9 is a
    // backdrop nobody chose.
    //
    // `textures` is the same thing said as a LIST, for a channel that owns
    // more than one stage and lets the user say which of them a given
    // thumbnail is shot on. The loader normalises both forms to a list, so
    // everything downstream reads `textures` and a one-backdrop channel is
    // simply a list of one.
    //
    // Which entry a frame is drawn on is the FRAME's (see state.bgFor) and
    // not the channel's — the pack states the set, the user cycles through it
    // with a button, and a channel offering only one backdrop has no button
    // because there is nothing to cycle to. Per frame rather than per run for
    // the reason the gradient and the flips are: it is an answer to what is
    // standing on the stage in this one picture.
    background: null,

    // A gradient laid over the backdrop AND over the figures standing on it,
    // and under the brand marks and the title.
    //
    // Over the figures is the whole point, and it is what separates this from
    // a backdrop with a dark foot painted into it: a gradient UNDER the
    // subject leaves them lit from the ankles up and floating, where the same
    // shade falling across their legs makes the pair read as one photograph in
    // one light. Under the marks and the type because those two have to stay
    // legible whatever is behind them.
    //
    //   heightRatio how far up the canvas it reaches, as a fraction.
    //   stops       [{at, color}, …], `at` running 0 at the BOTTOM edge to 1
    //               at the top of its own span — the direction the shade is
    //               thrown in. Colours carry their own alpha, so a scrim ends
    //               by fading to a fully transparent stop rather than by being
    //               given an opacity of its own.
    //
    // The direction is not a parameter, and that is the point rather than an
    // omission. A scrim is a rectangle, so only a ramp running straight up it
    // can run OUT along its own top edge; aimed at any angle, that edge would
    // cross the ramp instead and the shade would end on a visible horizontal
    // line. A channel wanting a ramp at an angle wants `photo.backdrop` or
    // `photo.tint` above, which are painted on shapes that have somewhere to
    // end.
    scrim: null,

    // A fixed brand mark, or a LIST of them — a channel with a logo in one
    // corner and a badge in another states both, and the loader normalises
    // the single form to a list of one.
    //
    // `left`/`top` are its inset from the canvas corner in canvas pixels, and
    // they are exact: this is a logo lock-up, so "10px from the top and 10px
    // from the left" is a specification and not a suggestion. `right` is the
    // same measurement taken from the other edge, for a mark that belongs to
    // the right-hand corner — stated as its own key rather than as a
    // computed `left`, because a mark hung off the right edge has to STAY
    // hung off it whatever width the file turns out to be, and a `left` is a
    // number that silently stops meaning what it said the moment the artwork
    // is re-exported a few pixels wider. `right` wins when both are given.
    //
    // Drawn at the image's own size unless `width` says otherwise, and
    // `shadow` is the one it casts — which, like the cutout's, follows the
    // image's ALPHA rather than its bounding box, so a badge cut to an
    // irregular edge throws that edge and not a rectangle.
    //
    // Not movable and not deletable, unlike an image the user uploads. It is
    // part of the channel in the way the backdrop is, and a brand mark that
    // can be dragged off the canvas is one that will be.
    stamp: null,

    // The subject, cut out of the photo by the backend (see /cutout-frame) and
    // placed on the backdrop. Where the user has since dragged them lives on
    // the frame (see state.cutoutFor), exactly as a dragged title does.
    //
    // Every key here describes where the subject ARRIVES. None of them
    // describes where they may go, and that is deliberate: a channel proposes
    // a composition, and the person using it is the one composing. The subject
    // can be dragged anywhere on the canvas and zoomed well past their starting
    // size; the only thing they cannot do is leave it entirely (see
    // compose.clampCutout).
    //
    //   fitWidthRatio  the share of the canvas width they are fitted into on
    //                  arrival. 0.5 stands them in the left half, leaving the
    //                  other half for the title.
    //   anchorX        where that room sits across the canvas, as the x of its
    //                  centre in the same 0-to-1 fraction `framing.anchorX`
    //                  uses. It moves the room and never resizes it, which is
    //                  the whole reason it is a second key: how big the subject
    //                  arrives is `fitWidthRatio` and stays `fitWidthRatio`.
    //                  Absent, the room hugs the left edge — the placement
    //                  every cutout channel had before this existed, where the
    //                  subject takes one half and the title the other. A
    //                  channel whose subject stands in the MIDDLE of something,
    //                  with the type above them rather than beside them, states
    //                  0.5 and keeps the size it already had.
    //   fitRatio       how much of that room they actually take, so the
    //                  placement reads as a person standing in their half
    //                  rather than one wedged into it.
    //   bottomRatio    where their feet land, as a fraction of canvas height.
    //                  1 stands them on the bottom edge.
    //   minZoom/maxZoom the ends of the zoom control, as multiples of that
    //                  arrival size.
    //   shadow         cast by the cutout's own silhouette, since it has an
    //                  alpha channel and fabric shadows follow it.
    //   overTitle      whether the figures are drawn OVER the caption instead
    //                  of under it. The stacking order of a channel's own
    //                  furniture, and the one piece of it a pack has ever had
    //                  a reason to state: everywhere else the type is the last
    //                  thing down because it is the thing that has to stay
    //                  readable, and a channel that composes the subject and
    //                  the words into one picture — words behind a shoulder,
    //                  a head breaking the line above it — wants the opposite.
    //
    //                  It is a real trade and not a preference. Under the
    //                  figures the caption can be covered by them, and nothing
    //                  in this app stops that happening: the subject arrives
    //                  where the channel put them and the user moves both, so
    //                  a channel asking for this is one whose composition
    //                  keeps them apart — the type at one end of the frame and
    //                  the figures at the other — and is trusting the overlap
    //                  to be the exception that makes the picture.
    //
    //                  What it does NOT change is which of them a click lands
    //                  on. A press already tests the figures before a movable
    //                  title (see the press handler in reframe.js), because
    //                  the subject is the thing being composed; that order was
    //                  this one before there was a key for it.
    //
    // A slot can also show the subject more than once — the same person
    // overlapping themselves, which is a look nothing else in the app can
    // produce and which a channel offers on some of its twenty options rather
    // than on all of them:
    //
    //   copiesBySlot   how many figures each grid slot gets, indexed by slot.
    //                  A property of the SLOT and not of the photo in it, so
    //                  the same option is the same arrangement whatever video
    //                  is loaded. Slots past the end take the last entry.
    //                  Every figure past the first is cut from a DIFFERENT
    //                  moment of the same shot — see vary.alternate_moments.
    //   copyScale      each figure BEHIND the front one, as a fraction of the
    //                  one in front of it. 1 for figures that are already
    //                  different photographs and need no depth cue to be told
    //                  apart.
    //   copyOffsetGap  ...and how far to the side it stands, as a fraction of
    //                  how wide the two of them are drawn. A fraction rather
    //                  than a pixel count because the figures are not a fixed
    //                  size: the same number has to hold a full-length shot
    //                  and a head-and-shoulders apart by the same amount OF
    //                  THEM, and a count of pixels that reads as two people
    //                  standing side by side in one case reads as one person
    //                  with a sliver showing behind them in the other.
    //   copyOffsetPx   the same thing said in canvas pixels, for a channel
    //                  that wants a fixed step. Ignored when copyOffsetGap is
    //                  set.
    //                  All of these describe how the figures ARRIVE and
    //                  nothing else — every one of them is draggable,
    //                  resizable and reorderable on its own afterwards. They
    //                  arrive spread rather than stacked because the point of
    //                  showing the subject more than once is that you can SEE
    //                  more than one of them: figures landing on top of each
    //                  other read as one figure with a rendering fault, and
    //                  the user has no reason to think there is anything
    //                  behind it to drag out.
    //
    // And `glow`, which is the one thing here that is not about placement at
    // all: a band of light lying INSIDE the silhouette and drawn OVER the
    // person. It is the cut edge made deliberate — a figure lifted off a
    // photograph has a hard outline whether anyone wants one or not, and this
    // turns it from an artefact into the reason they read as standing in front
    // of the stage rather than pasted onto it.
    //
    // Two properties define it, and everything in compose.glowSprite exists to
    // hold them.
    //
    // It never puts a pixel OUTSIDE the cutout. The silhouette the backend cut
    // is the silhouette, lit or not: any colour past the outline is a margin
    // between the person and the stage where the photograph had none, and the
    // eye reads that margin as the join.
    //
    // And it FADES INWARD. Depth is measured perpendicular to the outline,
    // going into the body — so the band is at full strength at the pixel
    // touching the cut edge and falls away monotonically from there, reaching
    // nothing at `edgeRatio`. The brightest pixel is always the one on the
    // edge. That is what makes it light catching an edge rather than a stripe
    // of paint: a band of even alpha has a second edge, the inner one, and a
    // second edge is the thing that reads as drawn.
    //
    //   colors     what each step of the button is, IN ORDER, and the order is
    //              half the feature: it walks colors[0], colors[1], … and then
    //              off, and round again, so a channel adds a third simply by
    //              writing it. Off is the step past the last one; a frame
    //              arrives at colors[0], because a channel that states a glow
    //              has decided its figures are lit.
    //   labels     what to call each step on the button, in the same order.
    //              Stated rather than derived: "Contour 1" tells the user
    //              nothing and the hex tells them less.
    //   edgeRatio  how far the band reaches inward, as a fraction of the
    //              figure's own longer side. A fraction, like everything in
    //              `cutout`, because the light is a graphic treatment OF the
    //              figure and has to hold its proportion as the figure is
    //              resized — in canvas pixels it survives the zoom control as
    //              a heavy collar on a small subject and a hairline on a large
    //              one.
    //   edgeAlpha  the opacity AT THE OUTLINE, and literally so: the band is
    //              normalised to peak at 1 before its falloff is applied, so
    //              this number is not quietly scaled by the shape of the fade.
    //   edgeFalloff how steeply it fades inward. 1 is the fade the
    //              construction gives on its own; above 1 is more aggressive,
    //              the outline keeping almost all its strength while
    //              everything behind it drops away fast. Separate from
    //              `edgeRatio` because that is HOW FAR the light reaches and
    //              this is HOW FAST it goes; neither can be had from the
    //              other, and because of the normalisation above it does not
    //              disturb `edgeAlpha` either.
    //   blend      how the band meets the person under it — any canvas
    //              composite operation, "screen" by default. Screen can only
    //              lighten, and least where what is underneath is already
    //              bright, so the light bites on a dark shoulder and fades
    //              across a highlight instead of laying one flat tone over
    //              both. It cannot darken anything, so there is no way for it
    //              to leave a seam.
    //
    //              It does come with a constraint on `colors`, and it is worth
    //              knowing before picking one. Screen leaves the photograph
    //              underneath with 1 - alpha*channel/255 of its own variation,
    //              per channel — so a BRIGHT glow colour erases the picture it
    //              is lying on, and a channel at 255 erases that channel
    //              outright, since screen(anything, 255) is 255. Under screen,
    //              how luminous the light looks and how much of the subject
    //              survives it are one dial turned in opposite directions.
    //
    // The band is taken from the cutout's own alpha, so it follows hair and
    // fingers rather than a rectangle, and the mask is hardened first — a
    // segmentation fringe is soft over about nine pixels, which is enough to
    // spend the ramp inside the slope and wash the band out.
    //
    // Which colour a frame is wearing is the FRAME's (see state.figureGlowFor)
    // and applies to every figure on it at once. Per frame because it answers
    // what is behind this one subject; all figures together because they are
    // one subject shown three times, and three people in three different
    // colours is not a look anybody asked for.
    //
    // Drawn after the figure it belongs to rather than after all of them, so
    // that a figure standing in front of another covers that one's light (see
    // compose.addFigureGlow).
    cutout: null,
};

/**
 * How this channel wants the video's frames CHOSEN, as opposed to drawn.
 *
 * Null for every channel that is happy with the pipeline's own answer. A
 * channel that fills it in is sent to the backend when a run starts (see
 * main.processVideo and backend/framing.py), where every number is clamped
 * before it is used.
 *
 * Which means, unlike everything else in a pack, this one is read at a moment
 * the user has to reach BEFORE the frames exist: the channel picked above the
 * link box is the one a run is framed for. Changing a frame's channel
 * afterwards changes how that frame is drawn, not which moment of the video it
 * was cut from — there is nothing left to re-choose by then.
 */
const BASE_FRAMING = {
    faceAreaRatio: null,   // share of the canvas the face fills when it is alone in shot
    anchorX: null,         // where the face is placed, as a fraction of the canvas
    anchorY: null,
    minTextSpace: null,    // width that must stay clear beside the face, or the frame is unusable
    weights: null,         // partial override of the scoring criteria — see quality_scorer.weights
};

/**
 * What a channel that FETCHES FRAMES rather than making thumbnails says about
 * itself. Absent from every channel that makes a thumbnail the ordinary way,
 * which is all of them but one.
 *
 * A capture channel turns the app into a frame fetcher: a video goes in and a
 * numbered grid of stills comes out, chosen on picture quality and spread
 * across the running time, with no title, no cutout and no brand furniture
 * anywhere. `frames.default` is what the count slider opens on, and
 * `frames.min`/`frames.max` are its two ends — the default is where the
 * handle STARTS and not a floor under it, so a pack may open on forty and
 * still let one be asked for; `output` is the size each still is
 * delivered at, and is the one number both sides of the app have to agree on
 * (see config.setCanvasSize and framing.canvas_w on the backend).
 *
 * Read at the same moment `framing` is — once, from the picker above the link
 * box, as a run starts. It cannot be a per-frame answer for a stronger reason
 * than framing's: it decides what the run IS.
 */
const BASE_CAPTURE = {
    frames: { default: 40, min: 1, max: 200 },
    output: { width: 540, height: 960 },
    // Whether each still comes back with the sentence that was being said
    // over it, ready to be set as its title (see backend/transcriber.py).
    //
    // A property of the capture block rather than of `title`, because it is
    // not a question about type: it decides whether the RUN transcribes the
    // video at all, which is read once as the run starts, in the same breath
    // as the frame count and the output size. A channel that draws no words
    // on its frames would be paying for a transcription it then discards.
    //
    // Off by default, so a capture channel that says nothing about it behaves
    // exactly as the only one that existed before this did.
    captions: false,
    // Whether those captions are REWRITTEN before they are set: the same
    // words, on the same timeline, turned from speech into third-person
    // report (see backend/narrator.py).
    //
    // Beside `captions` rather than inside `title` for the reason `captions`
    // is: it is not a question about type. It decides what the RUN does with
    // the video, and it is read once as the run starts along with the frame
    // count and the output size.
    //
    // Meaningless without `captions`, and the backend ignores it without one
    // rather than transcribing a video nothing asked it to transcribe.
    //
    // Off by default, so the pack that captions its frames and says nothing
    // about this goes on behaving exactly as it did before it existed.
    rewrite: false,
};

/**
 * Brand defaults that aren't type: the dark gradient behind the title, and
 * which side of the frame the title and the subject each take. Per-channel
 * looks that the user can still override per frame — see
 * editor.applyBrandingDefaults for how a frame the user has decided for
 * themselves stops taking the brand's answer.
 *
 * `flipText` moves the title block to the right. `flipImage` MIRRORS the
 * photo, which is how the subject ends up on the left: the automatic framing
 * anchors every face at the right-hand rule-of-thirds line and selects frames
 * on having clear space to its left (reframe_engine.RULE_OF_THIRDS_X,
 * MIN_TEXT_SPACE_FRAC), so left-of-frame subjects are not something the
 * backend produces — mirroring is what puts one there. The cost is that
 * anything legible in the photo comes out reversed; the Flip image button
 * turns it back off for a frame where that shows.
 */
const BASE_BRANDING = {
    gradient: true,
    flipImage: false,
    flipText: false,
};

// ── Merging packs onto the skeleton ───────────────────────────────────────

/**
 * Keys a pack uses to talk to the loader rather than to the renderer.
 *
 * `_`-prefixed keys are how a JSON file carries the prose that every other
 * file in this project carries as comments — the format has nowhere else to
 * put it, and a channel definition that cannot say why it is the way it is
 * would be the one thing here nobody could maintain.
 */
const isMeta = (k) => k.startsWith("_") || k === "schema" || k === "use";

/** Plain-object deep merge, so a channel overrides one nested value without restating its siblings. */
function merge(base, over) {
    if (!over) return { ...base };
    const out = { ...base };
    for (const [k, v] of Object.entries(over)) {
        if (isMeta(k)) continue;
        out[k] = (v && typeof v === "object" && !Array.isArray(v) && base[k] && typeof base[k] === "object")
            ? merge(base[k], v)
            : v;
    }
    return out;
}

/** Strips the loader's own keys from an object that has no skeleton to merge onto. */
const clean = (obj) => merge({}, obj);

/**
 * As `clean`, for the two keys the SERVER adds rather than the pack author.
 *
 * `id` and `base` are how a face or a preset says which one it is and where
 * its files are; both are answered here and neither means anything to the
 * renderer. Dropped so that what ends up on a channel is only what someone
 * wrote in a pack — a style object with a stray `base: "content/presets"` on
 * it is the kind of thing that reads as significant six months later.
 */
function withoutServerKeys(obj) {
    const { id, base, ...rest } = clean(obj);
    return rest;
}

/**
 * Expands `{"use": "<preset-id>", ...overrides}` anywhere in a pack.
 *
 * This is what lets two channels share one definition of a design — the black
 * title slab, the Karens halo — instead of holding two copies of it that
 * drift the first time only one is edited. Recursive, so a preset may itself
 * be written in terms of another.
 */
function expand(value, presets, seen = new Set()) {
    // An array is expanded THROUGH rather than returned as it stands: a list
    // of objects — the brand marks a channel locks to its corners, say — is a
    // list whose entries are as entitled to say `use` as any other object in a
    // pack. Returning it untouched made `{"use": ...}` mean nothing inside a
    // list and everything outside one, which is a rule nobody would guess.
    if (Array.isArray(value)) return value.map(v => expand(v, presets, seen));
    if (!value || typeof value !== "object") return value;

    let out = {};
    for (const [k, v] of Object.entries(value)) {
        if (isMeta(k)) continue;
        out[k] = expand(v, presets, seen);
    }

    const ref = value.use;
    if (ref) {
        if (seen.has(ref)) {
            warn(`the preset "${ref}" is defined in terms of itself — ignoring the cycle`);
            return out;
        }
        const preset = presets[ref];
        if (!preset) {
            warn(`the preset "${ref}" does not exist, so whatever it was standing in for is missing`);
            return out;
        }
        out = merge(expand(preset, presets, new Set([...seen, ref])), out);
    }
    return out;
}

// ── Asset paths ───────────────────────────────────────────────────────────

/**
 * Turns a path written in a pack into one the page can fetch.
 *
 * A bare path is relative to the frontend root — `fonts/Anton-Regular.ttf`,
 * `textures/foo.png` — which is what every such path in this app has always
 * meant, and what keeps the shipped faces where the stylesheet, the installer
 * and the launcher's preflight already expect them.
 *
 * `./something` is relative to the pack's own folder instead, which is what a
 * self-contained channel dropped in by hand uses: it can carry its font and
 * its foil next to its channel.json and name them without knowing where it
 * was installed. An absolute URL is left alone.
 */
function assetUrl(path, packBase) {
    if (!path) return path;
    if (/^([a-z]+:)?\/\//i.test(path) || path.startsWith("data:")) return path;
    const rel = path.startsWith("./") ? `${packBase}/${path.slice(2)}` : path;
    // Spaces and the like are encoded here rather than being left to the
    // browser, so the URL that gets requested is the same one whatever the
    // file happens to be called. The separators are kept as separators.
    return rel.split("/").map(encodeURIComponent).join("/");
}

/**
 * Makes a pack-declared face available to the canvas.
 *
 * A face with `"css": true` is already declared in styles.css, with local()
 * sources after the url() as a second route to the same file — a stronger
 * declaration than this one, so it is left alone. Everything else is
 * registered here from its own JSON, which is the mechanism that lets a new
 * channel arrive with a font nobody has edited a stylesheet for.
 *
 * Registration is fire-and-forget: FontFace.load() is not awaited, because
 * fontguard is what actually decides whether a face arrived, and it is asked
 * that question again for every channel as it is picked.
 *
 * `stretch` and `variationSettings` are how a VARIABLE font is pinned to the
 * one instance a channel is set in. A variable file holds a whole family
 * along its axes and, asked for by name alone, hands back its default — the
 * Regular at normal width, not the Condensed Bold the brand is set in. A
 * descriptor with a single value is not a filter but a clamp: whatever the
 * canvas asks for is pulled into the range declared here, so the face can
 * only ever be drawn at that instance. Both are stated for a pinned face:
 * `stretch` is the standard route and is what does the work, and
 * `variationSettings` says the same thing to the axes directly, so the
 * instance survives a browser whose font matching ever stops clamping.
 */
const unquoted = (name) => String(name || "").replace(/^\s*["']|["']\s*$/g, "");

function registerFace(face) {
    if (face.css || !face.file || typeof FontFace === "undefined") return;
    // Compared unquoted, because a FontFace reports back the CSS
    // serialisation of its family and a name with a space in it comes back
    // wrapped in quotes it was not given. A check that missed that would
    // register the same face again on every call — harmless, but it would
    // also make "is this face already here" unanswerable.
    if (document.fonts && [...document.fonts].some(f => unquoted(f.family) === unquoted(face.family))) return;
    try {
        const sources = [`url("${face.file}")` + (face.format ? ` format("${face.format}")` : "")]
            .concat((face.local || []).map(n => `local("${n}")`));
        const ff = new FontFace(face.family, sources.join(", "), {
            weight: face.weight || "normal",
            ...(face.stretch ? { stretch: face.stretch } : {}),
            ...(face.variationSettings ? { variationSettings: face.variationSettings } : {}),
        });
        document.fonts.add(ff);
        ff.load().catch(() => {});
    } catch (e) {
        warn(`the face "${face.family}" could not be registered (${e.message})`);
    }
}

// ── Loading ───────────────────────────────────────────────────────────────

const _problems = [];
const warn = (text) => { _problems.push(text); console.warn(`[channels] ${text}`); };

/**
 * Everything that went wrong while loading content.
 *
 * Collected rather than thrown, and surfaced by the UI: a pack with a typo in
 * it should cost the app that channel, not the page — but silently dropping a
 * channel the user is looking for is how a broken pack goes unnoticed for a
 * week. See ui.reportContentProblems.
 */
export const contentProblems = () => _problems.slice();

let CHANNELS = [];

/**
 * Reads one pack into a finished channel.
 *
 * `pack.base` is the URL its own folder is served from, so a `./` path in it
 * can be resolved; `faces` and `presets` are the merged maps from the same
 * document.
 */
function buildChannel(pack, faces, presets) {
    // `|| {}` because a pack is not obliged to set type at all. Every channel
    // that makes thumbnails does, and did when this function was written; a
    // channel that FETCHES FRAMES has no title, no face and no highlight,
    // because nothing is ever drawn over the picture (see `capture` above).
    // The skeleton underneath answers for all of it, and none of those answers
    // is ever reached — but they have to exist, because titleStyle() is
    // documented never to hand a caller a null style.
    const title = expand(pack.title, presets) || {};
    const branding = expand(pack.branding, presets);

    // A face is named — every channel on the same font then shares one
    // definition of it — or written inline, for a one-off.
    //
    // One function because a channel can now be set in two faces: the block's
    // and, on a channel whose highlight changes voice rather than colour, the
    // picked lines' (see `highlight.face` above). Resolved the same way or
    // they are not the same feature — a named highlight face that silently
    // failed to register would draw in the fallback while every measurement
    // agreed with it.
    const resolveFace = (ref, where) => {
        if (typeof ref === "string") {
            const named = faces[ref];
            if (!named) {
                warn(`${where} is set in the face "${ref}", which no pack defines. `
                     + "It falls back to the bundled face, so its titles will be laid out wrong.");
            }
            return named || faces[BUILTIN_FACE.id] || BUILTIN_FACE;
        }
        if (ref) {
            const inline = { ...clean(ref), file: assetUrl(ref.file, pack.base) };
            registerFace(inline);
            return inline;
        }
        return null;
    };

    const face = resolveFace(title.face, `the channel "${pack.id}"`);

    // The face REPLACES the skeleton's rather than merging onto it, which is
    // the one place in this loader where that is true. A face is an identity,
    // not a set of tweaks: merged, a face that leaves out a key inherits the
    // bundled face's answer to it, and the keys it is most natural to leave
    // out are `file` and `css`. A pack declaring a face it expects the machine
    // to already have would inherit `file: "fonts/TradeGothic...otf"` and
    // register ITS name against THAT file — a family that draws in somebody
    // else's letterforms, silently, with every measurement agreeing.
    const style = merge(BASE_TITLE_STYLE, { ...title, face: undefined });
    style.face = face || BASE_TITLE_STYLE.face;
    if (style.highlight) {
        // The highlight's own face is held out of the merge for the same
        // reason the block's is, and it matters more here: the skeleton's
        // answer is `null`, so a merge would leave a pack that names one with
        // whatever the previous key order happened to produce rather than
        // with a face. Resolved from the PACK's statement, not from the
        // merged style, so an inline face is registered exactly once.
        const hlFace = resolveFace((title.highlight || {}).face,
                                   `the channel "${pack.id}"'s highlight`);
        style.highlight = {
            ...style.highlight,
            face: hlFace,
            ...(style.highlight.texture
                ? { texture: assetUrl(style.highlight.texture, pack.base) }
                : {}),
        };
    }

    const decor = merge(BASE_DECOR, {
        photo: expand(pack.photo, presets),
        background: expand(pack.background, presets),
        scrim: expand(pack.scrim, presets),
        stamp: expand(pack.stamp, presets),
        cutout: expand(pack.cutout, presets),
    });
    // The pictures a channel brings with it, resolved the same way the
    // highlight foil above is — a bare path from the frontend root, a "./" one
    // from the pack's own folder.
    //
    // Both keys are normalised to their plural form here, and this is the only
    // place that knows there is a singular one: a pack may write one backdrop
    // or several, one brand mark or several, and everything downstream reads a
    // list either way. Doing it at the door rather than at each of the four
    // call sites is what stops "does this channel have one or many" from being
    // a question the renderer, the grid, the download and the panel each have
    // to answer for themselves — and answer the same way.
    const url = (path) => assetUrl(path, pack.base);
    if (decor.background) {
        const sheets = decor.background.textures || [decor.background.texture];
        decor.background = { ...decor.background, textures: sheets.filter(Boolean).map(url) };
        delete decor.background.texture;
    }
    if (decor.stamp) {
        const marks = Array.isArray(decor.stamp) ? decor.stamp : [decor.stamp];
        decor.stamp = marks.map(mark => ({ ...mark, texture: url(mark.texture) }));
    }

    return {
        id: pack.id,
        name: pack.name || pack.id,
        // Which label this channel belongs to, for the heading it is listed
        // under in the picker (see editor.initChannelPicker). A pack's own
        // statement rather than a list kept somewhere central, so a new
        // channel still arrives as nothing but a folder — and so a channel
        // that changes hands is one line in one file.
        //
        // Empty for a channel that names no label. Those are listed loose, at
        // whatever point in the order they fall, rather than under a heading
        // invented for them: a group of one that exists only because the code
        // wanted every row to have a parent tells the user nothing.
        group: pack.group || "",
        // Whether the run's overlay and subtitle inpainters should touch this
        // channel's frames. Passed through as the pack wrote it — the backend
        // clamps and defaults it (see CleanupRequest), and a pack that says
        // nothing gets them, which is what every channel but the two cutout
        // ones wants.
        cleanup: pack.cleanup ? clean(pack.cleanup) : null,
        // How this channel wants candidates deduplicated. Passed through as
        // the pack wrote it — the backend clamps an unknown name back to its
        // own default (see SelectionRequest), and a pack that says nothing
        // gets the colour histogram every channel has always used.
        selection: pack.selection ? clean(pack.selection) : null,
        title: style,
        scrim: decor.scrim,
        branding: merge(BASE_BRANDING, branding),
        ...decor,
        framing: pack.framing ? merge(BASE_FRAMING, expand(pack.framing, presets)) : null,
        capture: pack.capture ? merge(BASE_CAPTURE, expand(pack.capture, presets)) : null,
    };
}

/**
 * Fetches the content document and turns it into the channel list.
 *
 * Failure is reported, never thrown: the app is still usable for framing and
 * exporting photos without a brand, and taking the whole page down over a
 * missing folder would be a worse answer than a banner naming the folder.
 */
async function load() {
    let doc;
    try {
        const res = await fetch(CONTENT_URL, { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        doc = await res.json();
    } catch (e) {
        warn(`${CONTENT_URL} could not be read (${e.message}), so there are no channels `
             + "and no titles can be added");
        return;
    }

    const faces = {};
    for (const [id, spec] of Object.entries(doc.faces || {})) {
        const face = { ...withoutServerKeys(spec), id, file: assetUrl(spec.file, spec.base) };
        faces[id] = face;
        registerFace(face);
    }
    // The bundled face answers for itself if no pack declared it, so a
    // content directory that has lost its faces/ still measures against
    // something real.
    if (!faces[BUILTIN_FACE.id]) faces[BUILTIN_FACE.id] = BUILTIN_FACE;

    const presets = {};
    for (const [id, spec] of Object.entries(doc.presets || {})) presets[id] = withoutServerKeys(spec);

    // Problems the server hit while reading the directories — an unparsable
    // channel.json, a pack whose id disagrees with its folder. It can see
    // them and the browser cannot, so it reports them through the document.
    for (const problem of doc.problems || []) warn(problem);

    CHANNELS = (doc.channels || [])
        .filter(p => p.enabled !== false)
        .map(p => buildChannel(p, faces, presets));

    if (!CHANNELS.length) warn("no channels were found in the content directory");
}

// Top-level await, so every module that imports this one — and therefore the
// whole app — starts with the channel list already populated. That is what
// keeps the API below synchronous: the alternative is making titleStyle()
// async and reworking every call site in text.js for a fetch that resolves
// off local disk before the first frame is ever drawn.
await load();

// ── Active selection ──────────────────────────────────────────────────────

/** The dropdown's blank default — no channel, no guidelines, no title drawn. */
export const NO_CHANNEL = "";

let _activeId = NO_CHANNEL;

export const channels = () => CHANNELS;
export const activeChannelId = () => _activeId;
export const activeChannel = () => CHANNELS.find(c => c.id === _activeId) || null;
export const hasChannel = () => !!activeChannel();

/**
 * Runs `fn` with `id` as the channel being drawn with, and puts back whatever
 * was active before — including when `fn` throws.
 *
 * This is what lets a page hold twenty frames on five different channels
 * while the renderer still asks a module-level question. `fn` MUST be
 * synchronous: the swap lasts exactly as long as the call, and an async body
 * would hand the rest of its work to a later turn, by which time the channel
 * has been put back. Every drawing path here is synchronous (see
 * addTextOverlay), and the one thing that is not — decoding a logo — happens
 * outside the call.
 */
export function usingChannel(id, fn) {
    const previous = _activeId;
    _activeId = CHANNELS.some(c => c.id === id) ? id : NO_CHANNEL;
    try {
        return fn();
    } finally {
        _activeId = previous;
    }
}

/** Returns true when the selection actually changed, so callers can skip a re-render. */
export function setActiveChannel(id) {
    const next = CHANNELS.some(c => c.id === id) ? id : NO_CHANNEL;
    if (next === _activeId) return false;
    _activeId = next;
    return true;
}

/**
 * The type guidelines to render with.
 *
 * Falls back to the shared skeleton when nothing is selected so the measuring
 * code in text.js always has a face to work with — it is never *drawn* with,
 * since addTextOverlay draws nothing without a channel, but a measurement
 * asked for early must not have to handle a null style.
 */
export const titleStyle = () => styleFor(_activeId);

/**
 * The guidelines of one NAMED channel, whichever one happens to be active.
 *
 * Everything that draws asks for the active channel's style, because drawing
 * happens inside usingChannel and the answer is always "the one being drawn
 * with". This is for the questions asked OUTSIDE that — what shape a frame's
 * stored title has, what its panel should look like — where the channel is
 * the frame's and there is nothing to swap in and out around a synchronous
 * call. An unknown id (including the empty selection) answers with the
 * skeleton, for the same reason titleStyle does: a caller measuring or
 * inspecting must never have to handle a null style.
 */
export const styleFor = (id) => (CHANNELS.find(c => c.id === id) || { title: BASE_TITLE_STYLE }).title;

/** The active channel's non-type brand defaults (see BASE_BRANDING). */
export const branding = () => (activeChannel() || { branding: BASE_BRANDING }).branding;

/**
 * The face measured against before any channel is picked — the page's own
 * title fit, and fontguard's default check.
 */
export const defaultFace = () => BASE_TITLE_STYLE.face;

/**
 * How a channel presents the photograph, its backdrop, its brand mark and its
 * cutout rules — all of them at once, because everything that draws them draws
 * them together and in this order (see compose.buildComposite).
 *
 * By id rather than off the active selection, for the same reason styleFor is:
 * the grid asks this about frames that are not the one on screen, and a
 * frame's furniture belongs to the brand IT was made under.
 *
 * Named key by key rather than spread from the channel, which is deliberate —
 * what comes back is decor and nothing else, so a caller cannot reach the
 * title through it. The cost is that a key added to BASE_DECOR and not added
 * here is a key every pack can state and nothing will ever read; it fails
 * silently, as the channel simply drawn the ordinary way.
 */
export const decorFor = (id) => {
    const c = CHANNELS.find(ch => ch.id === id);
    return c
        ? { photo: c.photo, background: c.background, scrim: c.scrim, stamp: c.stamp, cutout: c.cutout }
        : BASE_DECOR;
};

/** Whether channel `id` composes its thumbnails from a cutout rather than from the photo. */
export const usesCutout = (id) => !!decorFor(id).cutout;

/**
 * A channel's framing profile in the shape the backend takes, or null when it
 * has none and the pipeline's own numbers stand.
 *
 * The rename from the pack's camelCase to the request's snake_case happens
 * here and nowhere else. Two vocabularies is one more than anybody wants, but
 * each is idiomatic on its own side, and the alternative is a pack that reads
 * like Python or an API that reads like JavaScript — with the translation
 * still happening, just implicitly, in whichever file forgot.
 */
export function framingFor(id) {
    const c = CHANNELS.find(ch => ch.id === id);
    if (!c) return null;
    const f = c.framing || BASE_FRAMING;
    const out = {};
    const put = (key, value) => { if (value !== null && value !== undefined) out[key] = value; };
    put("face_area_ratio", f.faceAreaRatio);
    put("anchor_x", f.anchorX);
    put("anchor_y", f.anchorY);
    put("min_text_space", f.minTextSpace);
    put("weights", f.weights);
    // Whether a face that overfills the CROP costs the frame its score. A
    // channel that composites a cutout never shows the crop, so there the
    // penalty demotes exactly the frames it most wants — the close ones,
    // which carry the most resolution on the only thing that survives.
    put("face_fill", f.faceFill);
    // The canvas the frames are rendered onto travels with the framing rather
    // than beside it, because on the backend it IS framing: the cover scale,
    // the anchor and the overpan margin are all expressed against it, and one
    // profile per run is what stops those four numbers disagreeing (see
    // framing.py). A capture channel is the only one that states a size; every
    // other channel says nothing and gets the 1280x720 it always had.
    if (c.capture) {
        put("canvas_w", c.capture.output.width);
        put("canvas_h", c.capture.output.height);
    }
    return Object.keys(out).length ? out : null;
}

/**
 * Channel `id`'s capture block, or null for a channel that makes thumbnails.
 *
 * This is the question "is this run a frame fetch", asked by the count box
 * above the link, by the pipeline that sends the request, and by every control
 * that has nothing to do on a frame with no title and no brand.
 */
/**
 * Whether channel `id` wants the run's overlay and subtitle inpainters, in the
 * shape the backend takes — or null when it has no opinion and they run, which
 * is what every channel that composites the photograph wants.
 *
 * Read at the moment a run STARTS, like the framing and the capture block, and
 * for the same reason: these passes clean the frames as they are made, so by
 * the time a channel could be re-picked from the title panel the damage (or
 * the cleaning) is already in the pixels.
 */
/**
 * How channel `id` wants candidate frames deduplicated, in the shape the
 * backend takes — or null when it has no opinion and the colour histogram
 * stands, which is what every channel but the two cutout ones wants.
 *
 * Read at the moment a run STARTS, like the framing and the cleanup block:
 * this decides which twenty frames the video becomes, and by the time a
 * channel could be re-picked from the title panel they have already been
 * chosen.
 */
export const selectionFor = (id) => {
    const c = CHANNELS.find(ch => ch.id === id);
    return (c && c.selection) || null;
};

export const cleanupFor = (id) => {
    const c = CHANNELS.find(ch => ch.id === id);
    return (c && c.cleanup) || null;
};

export const captureFor = (id) => {
    const c = CHANNELS.find(ch => ch.id === id);
    return (c && c.capture) || null;
};
