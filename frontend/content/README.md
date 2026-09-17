# content/ — the parts of the app that are data, not code

Everything in here is read at run time and merged into the app. Nothing in
here is imported by a JavaScript module, which is the whole point: a new
channel is a folder that gets dropped in, not an edit to `channels.js` and a
rebuild of the installer.

```
content/
  faces/      one .json per type face the channels can be set in
  presets/    named fragments several channels share (a panel, a glow)
  channels/   one folder per channel, each with a channel.json
```

The frontend server aggregates all three into a single document at
`/content/channels.json`, and `js/channels.js` fetches that once at startup.

## Two roots, and which one wins

The server looks in two places and merges them:

| Root | What lives there |
|---|---|
| `<install>/app/frontend/content/` | what shipped — replaced wholesale by every patch |
| `<install>/content/` | what this machine added — never touched by an update |

Same layout in both. An id present in both comes from the second, so a machine
can override a shipped channel without its change being undone by the next
patch; and a channel added to the second root survives every update, because
updates never look at it.

In the repo (no install), the external root is `<repo>/content/` if it exists.

## Paths inside these files

A `file` or `texture` is resolved relative to the **frontend root**, so
`fonts/Anton-Regular.ttf` and `textures/foo.png` mean what they have always
meant. A path starting with `./` is resolved relative to the pack's own
folder instead, which is what a self-contained channel uses:

```
content/channels/new-channel/
  channel.json
  Face.otf          ->  "file": "./Face.otf"
  foil.png          ->  "texture": "./foil.png"
```

## Language

A channel's `name` is a brand and is whatever that channel is called, in
whatever language it is called it. Everything else visible anywhere in the app
is English — see CLAUDE.md, and `installer/check_language.py`, which fails the
build over it.

## Adding a channel

Copy an existing folder under `channels/`, change the `id` (it must match the
folder name), the `name`, and whatever makes it that channel. Everything left
out falls back to the shared skeleton in `js/channels.js` — a channel states
only what makes it different.

Reload the page. There is no build step and nothing to restart.

## What a channel can say

Most of a pack is `title` — the face, the colours, the layout, the highlight —
and every key in it is documented where the renderer reads it, in the shared
skeleton at the top of `js/channels.js`. A few things sit outside `title`, and
they are the ones that change what a thumbnail IS rather than how its type is
set:

| Key | |
|---|---|
| `photo` | the frame's own photograph drawn as a card — resized, rounded, tinted, on a ground the channel paints — see below |
| `background` | a picture drawn under everything, instead of the frame's photo |
| `scrim` | a shade thrown up from the bottom edge, over whatever is under it — see below |
| `stamp` | a brand mark locked to a corner, on every frame |
| `cutout` | the subject, cut out of the photo and placed on that background — including how many copies of them each grid slot shows |
| `framing` | how the video's frames are CHOSEN — see below |
| `capture` | the channel does not make thumbnails at all — see below |

All seven are absent from every channel that makes a thumbnail the ordinary
way: the photograph, cropped, with a title beside it. `laugh-society-1` is the
pack to read for what `background`, `scrim`, `stamp` and `cutout` look like
filled in, and it explains each of its own numbers where it states them.

### A channel may draw the photograph as a card

`photo` is the frame's own picture stopped from being the whole surface. It is
scaled down, centred, rounded at the corners and tinted, on a ground the
channel paints — and the ground only exists because the picture was shrunk, so
the four keys are one composition rather than four effects.

```json
"photo": {
    "fillRatio": 0.85,
    "radius": 13,
    "backdrop": { "from": [0, 1], "to": [1, 0], "stops": [ ... ] },
    "tint":     { "from": [0, 1], "to": [0, 0], "stops": [ ... ] }
}
```

| | |
|---|---|
| `fillRatio` | how big the card is, as a fraction of the canvas, applied to BOTH dimensions |
| `radius` | its corner rounding, in canvas pixels, clamped to half the card |
| `backdrop` | the gradient painted over the whole canvas under it — what the margin is |
| `tint` | a second gradient painted over the card and clipped to it, corners included |

`noisy-dish` is the pack that does it, and it explains each of its own numbers
where it states them.

Both gradients are written the way `title.sticker.fill` is: `from` and `to` are
corners as fractions of the box the ramp runs across, so `[0,0]` to `[0,1]` is
top to bottom and `[0,1]` to `[1,0]` is bottom-left corner to top-right, and
the stops are the same `{at, color}` pairs a scrim's are.
Colours carry their own alpha. Which box they are measured in is the whole of
the difference between them — the backdrop's is the canvas, the tint's is the
card — so the tint's `[0,1]` is the bottom edge of the *picture* and not the
bottom of the frame.

That is also what separates `tint` from `scrim`, which is otherwise the same
idea: a scrim lies across the canvas and would run out over the ground, where
a tint is a treatment OF the photograph and ends exactly where it does.

One `fillRatio` and not a width and a height, because the still arrives already
cut to the canvas's shape (see `capture.output`): scaling both by the same
fraction is the one move that leaves the picture the picture, where two numbers
would re-crop it or squash it to a shape nobody chose. Centred for a related
reason — a margin that is not equal on both sides is a composition, and a pack
should have to say that out loud rather than get it by arithmetic.

Everything above the card is still composed against the canvas. The figures,
the brand marks and the title do not know there is a card, which means a title
at the foot of the frame stands in the ground below it and grows up across it
as it gains lines. That is a placement to design for, not a bug to route
around: the margin is part of the look or the pack should not have asked for
one.

### A channel may be set in two faces

`title.face` is the block's, and it is the one every channel states. A channel
whose highlight changes *voice* rather than colour states a second one at
`title.highlight.face` — same two forms, a face id from `faces/` or a face
written inline. `laugh-society-2` is the pack that does it: the title is set in
a slanted face and a picked line in an upright one, so the picked line reads as
somebody else speaking rather than as the same voice under a light.

Two things follow, and both are worth knowing before reaching for it:

- Everything about a picked line is **measured in its own face** — the width
  fit, the ink bounds, the cap band a box is padded from. Two faces on one
  block means two sets of metrics, and the renderer keeps them apart (see
  `styleFor` in `js/text.js`). Both faces are loaded and verified before the
  channel is drawn with, and a missing one is named in the banner by itself.
- Under `uniformSize` the block takes **one** size, from whichever line needs
  the smallest one to reach the target width. A picked line measured in a wider
  face can therefore be what sets the size for the whole block, the lines in
  the block's own face included. That is what one size for a block means; it is
  not a bug to be fixed by giving the picked line a size of its own.

It works under either scope, and the renderer changes technique between them.
Under line scope a row is one string in one face. Under `scope: "word"` it is
cut into a run per treatment, each measured, stacked and drawn in its own face
at the pen the runs before it reached (`runRows` in `js/text.js`). The space
inside a run is still the font's; the one join arithmetic decides is the space
between two runs, which is a space between two faces and has no kerning pair to
lose. `hall-of-femme` is the pack that picks out words in a second face.

### A channel may hand some of its colours to the user

`title.colors` is a channel saying that some of its decisions are the user's,
per frame: what colour the letters are, what colour the words picked out of
them are, and what colour the slab behind them is. It is up to three lists, and
each becomes a row of swatches in the title panel.

```json
"colors": {
    "text":  ["#FF008A", "#FFE67C", "#FFFFFF", "#000000"],
    "panel": ["#FF008A", "#FFE67C", "#FFFFFF", "#000000"]
}
```

`hall-of-femme` is the pack that does it, and the same four colours in both
lists is the point rather than a repetition: what the show is set in is the
PAIRING, so sixteen combinations are offered instead of four cards.

The lists are as long as a pack wants them. Nothing counts them — the panel
draws a swatch per entry and the row wraps — so adding a colour to a show is
one string in one file. A pack states any of the three and leaves the rest out;
a row with no list behind it is hidden along with its label.

Five things are worth knowing before reaching for it.

- It is absent everywhere else, and that is the default this feature is
  measured against. A brand's colours are its colours; a picker in front of
  them is a picker in front of the guidelines. A channel offers a palette when
  the choice is part of the look, not to spare itself a decision.
- **`text` recolours the highlight too, unless `highlight` is stated.** A
  channel offering `text` alone is one whose type is a single colour the user
  picks — which is true where the highlight is a different CUT of the face
  rather than a different colour (see `highlight.face` above), and leaving it
  behind would mean choosing pink and getting one line still in white.
- **`highlight` says the picked words are their own question.** A channel whose
  highlight IS a colour, and whose colour is also the user's, states the list
  and `text` then stops reaching the highlight. Usually the same list twice
  over, because what is being offered is the second choice being a different
  answer to the first. `laugh-society-snapchat` is the pack that does it: a
  block in one of three colours, a picked word in another of the same three,
  set in a second face.
- **`panel` needs a `panel`, and `highlight` needs a `highlight`.** With no
  slab there is nothing to fill and with no highlight there are no picked words
  to recolour, so that row of buttons is simply not drawn rather than drawn
  dead.
- **A palette does not have to open with the colour the channel is set in.**
  A frame holds no colour of its own until somebody presses a swatch, so what
  it is drawn in is the pack's own `color`, `highlight.color` and `panel.fill`
  — and the swatch that shows as chosen is whichever one matches those. That is
  what lets every row be written in the same order while the channel still opens
  on white type and a hot pink card.

### `framing` is the one key with a deadline

Everything else in a pack is read at draw time, so switching a frame to another
channel redraws it in that brand and nothing is lost. `framing` is not: it
changes which moments of the video become thumbnails at all, and by the time
the title panel exists there are already twenty frames, chosen.

So it is read from the channel picked **above the link box**, once, when a run
starts — and a channel selected afterwards will still restyle the frame, but
cannot re-choose it. That is why there are two channel pickers on the page.

The numbers themselves are the pipeline's own defaults until a pack overrides
them, and every one of them is clamped on arrival; `backend/framing.py` is
where they live and what they mean.

### A caption may be set at a size the pack states

`title.layout.fontSize` is a number of canvas pixels, and it turns the size fit
off: every line is set at exactly that size.

It is `null` on every channel that makes a thumbnail, which is the right answer
for a headline — three or four words written to be a headline, on a block that
is one of the two things in the frame, set as large as the frame allows. It is
the wrong answer for a caption, whose length is not a decision anybody made: a
short sentence comes back enormous and a long one small, so a grid of forty
stills cut from one video is forty different type sizes. The three Snapchat
packs state `20`.

Three things follow, and all three are the point:

- `uniformSize` stops meaning anything — there is no per-line fit left to make
  uniform, so those packs drop the key.
- `maxBlockHeightRatio` stops pulling an overlong block in. A size the renderer
  may shrink is not a stated size, so a long caption makes a taller block and
  the user drags it.
- the line breaks come from width instead of height. `js/caption.js` normally
  breaks a sentence over the MOST lines that fit, because more lines meant
  bigger type; at a stated size it takes the fewest that fit the width, which is
  ordinary wrapping.

`title.layout.maxChars` is a second ceiling on those breaks, in characters. The
width asks what the frame holds; this asks what a viewer reads at a glance, and
a line has to be under both. The Snapchat packs state `20`.

### A background may be one box per line

`title.panel.scope` is `"block"` unless stated — one rectangle around the whole
title, which is what a panel has always been. `"line"` draws one per line, each
the width of its own line: the block stops being a card with type on it and
becomes a shape that follows the sentence, so a two-word line reads as a short
line instead of as a long line with air on both sides.

`title.panel.joined` is read under `"line"` only. False (absent) leaves the
boxes separate — a stack of tags with `lineGap` of photograph between them. True
overlaps them into one continuous background, stepping in and out with the
sentence but carrying a single outline and a single shadow. `hall-of-femme` does
both: `scope: "line"`, `joined: true`.

Three things follow besides the drawing:

- the boxes are what the rows are stacked by, so `bottomRatio` is the margin
  under the last box rather than under the last line. What `lineGap` measures
  depends on `joined`: separate, it is the gap between two boxes; joined, there
  is no such gap, so it is the gap between the lines themselves. A joined panel
  therefore wants a far smaller number — the Snapchat packs use `3`.
- every box takes one shared vertical band, so they differ in width and in
  nothing else — sized to their own rows they would come out at four heights
  over an accent and a descender.
- a box is sized to the whitespace the user typed as well as to the ink. A
  trailing space is ignored everywhere else in the renderer; behind a box it is
  how somebody asks for a gap — for an emoji to sit in, say. `js/state.js`
  stops trimming freeform lines for this reason.

### A background may have rounded corners

`title.highlight.radius` rounds the boxes behind picked words or lines;
`title.panel.radius` rounds the slab behind the whole block. Both are canvas
pixels, both default to `0` — a square corner, which is what every one of them
was until the Snapchat packs wanted 13.

Pixels rather than a fraction of the type, unlike `highlight.padRatio` beside
it, and the difference is deliberate. Padding is part of the type: it is the air
around the letters and has to grow with them. A corner radius is part of the
graphic, and one that scaled with the type would give a short caption soft
corners and a long one sharp. Both are clamped to half the box they round, so
overreaching gives a capsule rather than a shape that folds through itself.

### A caption may be hung from either end of the frame

`title.layout.vAlign` is `"center"` unless stated, which is where a headline
belongs: the photograph is composed around it and the block is one of the two
things in the frame. A caption is not that, and every Snapchat pack states
`"bottom"` instead — words laid over a picture that was composed without them
belong where the eye expects burned-in speech, and a vertical still has its
subject across the middle, so the two places a caption can go without covering
a face are the ends.

`"top"` is the other end, and it is the mirror of `"bottom"` rather than a
third idea. `bottomRatio` is the margin under the block and `topRatio` the
margin over it, both fractions of canvas height, and each is read only under
its own `vAlign`. The difference is what the margin buys: a block stood on the
bottom margin grows *upward*, which is why the renderer has to subtract the
block's own height to keep the distance constant as lines come and go; a block
hung from the top margin grows *downward*, so its first line lands on the
margin and stays there for nothing (`blockTop0` in `js/text.js`).

Both are measured to the first thing **drawn**, not to the letters. Under a
box highlight or a line-scope panel, a row with a background on it puts that
background's edge on the margin and the capitals a padding's worth inside it.

`conspiracy-central` is the pack that hangs its captions from the head of the
frame, 50px down, and it is also the one pack under the Snapchat heading with
no card behind its type at all — the two go together, and it says why.

### `capture` changes what the app IS

A pack with a `capture` block is not a brand, it is a mode. The app stops
making thumbnails and becomes a frame fetcher: a video goes in, and a numbered
grid of edited stills comes out, chosen on picture quality and spread across
the running time, in the order the moments occur in the video. There is no
brand mark and nothing to compose, so most of the side panel that exists for
those is simply not there — what is left around the canvas is zoom, the
downloads, and the text box if the pack asks for captions.

Every pack under the **Snapchat** heading does this, and none of them writes
the block out. They share one definition of it:

```json
"capture": { "use": "snapchat-capture" }
```

...which is `content/presets/snapchat-capture.json`:

```json
"frames": { "default": 40, "min": 1, "max": 200 },
"output": { "width": 540, "height": 960 },
"captions": true,
"rewrite": true
```

| | |
|---|---|
| `frames.default` | where the count slider above the link opens |
| `frames.min` / `frames.max` | the slider's two ends |
| `output` | the size every frame is delivered at |
| `captions` | each frame arrives with the sentence said over it |
| `rewrite` | ...and that sentence is reported rather than quoted |

Sharing it is not tidiness. A pack under this heading is made by copying the
folder next to it, so every key restated in each of them is a key a new channel
can silently be missing — which is exactly what happened: channels arrived
captioning their frames with the transcript as spoken, because the folder they
were copied from predated `rewrite`. One definition means a new channel inherits
the whole of it by writing one line.

A channel that ever wants the transcript as spoken says so in its own pack, by
overriding the key. Nothing does today.

A rewritten caption is written to stand on its own over one still: the model is
asked for a self-contained sentence inside 80 characters, and one that genuinely
needs more may run to 160 — which is then split across two consecutive frames
rather than repeated on both.

How many captions there are is decided by `frames`, not by the transcript. The
rewrite is told how many stills the run is about to deliver and writes that many
lines, so forty frames of a long video is an account in forty captions and two
hundred is the same account in two hundred. The count slider above the link box
is therefore a control over the writing as much as over the pictures.

Nothing repeats a caption; a frame with nothing left to say comes back with an
empty text box (`backend/transcriber.py`, `captions_for`). The long explanation
of what the rewrite does and what it refuses to do lives in the preset itself.

`output` is the one number both halves of the app have to agree on, and
neither holds a second copy of it: the backend is told it in the framing this
same pack sends (`js/channels.js`, `framingFor`), and the browser canvas is
set from it (`js/config.js`, `setCanvasSize`). Everything downstream of that
— the restoration, the edit presets, the zoom, the download — is the same code
every thumbnail goes through.

Read at the same moment `framing` is, and spent the same way: it decides what
the run produces, so it comes from the picker above the link box and cannot be
changed once the frames exist.

Choosing the frames is two stages, and the second is what makes the grid
usable rather than merely full. The first measures every sample and keeps a
spread of the sharp ones; the second goes back over that spread with the
rejected candidates still to hand and substitutes — a slot showing the same
scene as its neighbour is moved to the nearest following scene, and a slot
whose subject has their eyes shut is moved to the nearest moment of the same
scene where they are open, extracting a dense window of the video to find one
if the sample cannot offer it. See `backend/frame_grab.py`; neither pass ever
changes how many frames come back.

Variation works on these frames too, and means the same thing it means
everywhere else — another moment of this same shot — decided by the capture
rules rather than the face-matching ones (`vary.pick_capture_variation`).

Burned-in captions are removed, on this channel and on every other one. A
video's own text — the words across the middle of the picture, the agency
credit in the corner — is found once from a sample and painted out of every
frame that carries it, while a video with no burned-in text is left alone. It
runs on a 2 MB text-detection model fetched with the rest; without that model
it does nothing at all, which is deliberate. See `backend/subtitle_remover.py`
and `backend/text_detector.py`.

A capture pack usually wants `"minTextSpace": 0` in its `framing` and
`"gradient": false` in its `branding`. Both of those exist to make room for a
title beside the subject and to keep one legible across the whole frame, and a
captured still has neither — left at their defaults they would darken a corner
for nothing and reject frames for failing to leave space for a column of type
that is never drawn.

Unless the pack states a `scrim`, in which case `"gradient": true` is what it
wants — see below.

### `scrim` is the channel's own shade

`scrim` is a gradient the app draws itself, on the canvas, over the backdrop or
the photograph and under the brand marks and the title. `heightRatio` is how far
up the frame it reaches; `stops` are `{at, color}` with `at` running 0 at the
bottom edge to 1 at the top of the scrim's own span, so they read in the
direction the shade is thrown. Colours carry their own alpha, so a scrim ends on
a fully transparent stop rather than on an opacity of its own — otherwise there
is a line across the picture where it stops.

It runs straight up and there is no way to aim it, which is the same fact said
twice: a scrim is a rectangle, so only a vertical ramp runs out along its own
top edge instead of being cut off by it. A channel wanting a ramp at an angle
wants `photo.backdrop` or `photo.tint`, which are painted on shapes that have
somewhere to end.

Two packs use it for two different jobs. `laugh-society-2` throws it over a
composed stage so the cut-out figures and the backdrop read as one photograph in
one light. `laugh-society-snapchat` throws it over the bottom third of a
captured still because its caption is set straight onto the picture with no card
under it, and the shade is what makes that legible.

**A scrim is what the Gradient button means.** That button is one idea — whatever
the channel puts between the photograph and the title — and where it comes from
depends on the channel: the backend bakes a dark ramp into the photo it returns
for an ordinary thumbnail channel, and a channel with a scrim draws its own here
instead. A pack with a scrim states `"gradient": true` if it wants the shade on
arrival, and the button switches it with no round trip.

The two are never both applied, and neither is asked for on a channel that has
already decided the question another way. `state.backendPresentation` reports
`gradient` false to the backend for any channel with a `scrim` or a `photo`, and
`editor.syncGradientButton` hides the button entirely on a channel where the
backend's ramp would land on nothing a viewer can see — one composing from a
cutout, or one that has shrunk the photograph onto a ground of its own.

### `capture.captions` writes on the frames

With `"captions": true`, the run also transcribes the video's voice track and
each still comes back carrying the sentence that was being said at the moment
it was cut from. The frames already knew when they happened — every one of
them records its own second of the video — so all this adds is the words
knowing the same thing, and the two are crossed on the backend. See
`backend/transcriber.py`, which also explains how a stream of recognised words
is put back together into sentences that can be read on their own.

The words arrive as a starting point, not as a finished title. The frame's
text box is pre-filled with them and everything after that is the ordinary
title machinery — the same renderer, the same per-frame storage, the same
drag, the same download. So a pack asking for captions also has to say how
they are set, in an ordinary `title` block. The Snapchat packs say it once,
in `content/presets/snapchat-caption.json`, and that file is the worked
example: `"freeform": true` (the caption is a sentence, and its breaks are the
user's), `"fontSize": 40` (one stated size, so forty stills cut from one video
are one set rather than forty type sizes), `"maxChars": 20`, `"vAlign":
"bottom"` (burned-in speech sits at the foot of the frame — one pack hangs it
from the head instead, see below) and `"movable": true` (so it can be pulled
off a face).

Where the sentence breaks when it first arrives is decided in `js/caption.js`.
At a stated `fontSize` it takes the fewest lines that fit both the block's
width and `maxChars` — ordinary wrapping. Only a channel that leaves the size
to be fitted breaks by height instead, taking the most lines that fit
`maxBlockHeightRatio`, because there more lines meant larger type.

Nothing about it is fatal. A silent video, a machine that has never fetched
the speech model, a frame cut from a pause — each of those is a frame with an
empty text box, which is exactly what a capture run produced before captions
existed.

Captions also change what a new frame INHERITS, and this is the one place the
key reaches outside the pack. An untouched frame normally starts from its
nearest worked-on neighbour — its title, its highlight, where the block was
dragged (`state.carryOverSettings`), because a title is work somebody did and
the next frame is better off starting from it. None of that is true of a
caption: every frame already holds the one sentence that is right for it, so
inheriting the neighbour's is not a head start, it is the frame losing its own
words the moment it is clicked. Under `captions`, an image overlay is the only
thing that still travels between frames.

The caption comes out in whatever language was spoken. That is not an
exception to the rule below: it is the video's own words, which are content,
not an interface string. Under `rewrite` below that stops being merely what
happens and becomes something the app enforces.

### `capture.rewrite` reports them instead of quoting them

`captions` alone gives a frame the transcript. That is right for the half of
these videos that is narration, and wrong for the half that is interview: an
interview is somebody talking about themselves to a reporter who is not on
screen, and a still captioned "I am Brian, Dolly's son" is a caption of
nobody. The viewer has no previous card to learn who Brian is from, because
each frame is looked at alone — the same reason transcriber.py assembles
whole sentences in the first place.

With `"rewrite": true` the transcript goes through a language model before any
frame is captioned from it, and comes back as third-person report. The two
lines above become one that stands on its own: "The world learned of her death
through a message from her son Brian." See `backend/narrator.py`.

**One caption per thought, not per sentence.** The transcript is gathered
into thoughts before the model is shown any of it. A speaker introducing
themselves, then saying what they came to say, is three transcript sentences
and one piece of news — so those three arrive as one item and come back as one
caption. The grouping is arithmetic, not judgement: a sentence that never
finished, or one whose neighbour starts in lower case, is a sentence still in
the middle of a thought. That is `transcriber`'s own continuation rule run a
second time with its length brake off — the brake is there so a CAPTION fits
four lines of type, and here it is the model's answer that has to fit, not the
material it reads.

Doing it this way rather than asking was measured. Offered a syntax for
joining lines itself, the model used it zero times across sixty-three
sentences; told in a rewritten prompt that joining was the normal move, with a
worked example, it used it zero times again.

**It does not repeat itself.** A model with a numbered list to fill and
nothing new to say will paraphrase rather than leave a line out, which on a
real run produced three consecutive captions that were the same sentence with
its last three words swapped. Grouping is most of the cure, since there are
fewer lines to pad; a check drops any caption that reuses three quarters of
the wording of the one before it, and those frames keep the sentences as
spoken instead.

**It does not name the same person over and over.** Left alone, the rewrite
opens caption after caption with the same full name — "Dolly Parton was born",
"Dolly Parton's family", "Dolly Parton picked up a guitar" — because each
caption is shown alone and naming the subject in full is the safest way to
make a line stand by itself. Asking the model to vary it was measured twice,
the second time with the rule in capitals, and did not move: seven of
seventeen captions still opened with the full name, six of them consecutive.

So the second and later captions of a run are rewritten by
`backend/reference.py`, which cycles through the surname, the given name, a
pronoun and the person's role. Two of those are conditional and neither is
guessed: the pronoun appears only where the captions themselves said "her
mother" or "his aunt", and the role only where the text contained an
appositive to lift it from ("Dolly Parton, the singer" gives "The singer").
The swap fires only while the same person opened the caption before, which is
exactly the condition that makes a pronoun unambiguous — a second person named
in between puts the full name back.

**It does not move the timeline.** The model is never asked when anything
happened. It is handed the sentences numbered, and asked for text keyed to
those numbers; the seconds each caption covers are the ones the transcript
already carried, and a merged line takes the first sentence's start and the
last one's end. So a rewrite can come back worse than the transcript, but it
cannot come back describing a different moment of the video — which is the
failure that would matter, because it is invisible in review.

It rewrites; it does not research. Every name and fact in a caption has to be
somewhere in the transcript already. Nothing enforces that but the prompt,
which is worth knowing rather than assuming.

**It does not translate, and that took work.** A caption is the video's own
words put back over the video's own pictures, so a Portuguese interview
reported in English is captioning something nobody said — and unlike every
other way a rewrite can go wrong, it does not look wrong: it reads perfectly,
and nothing in a grid of forty frames invites a second look. Asking the model
for "the same language as the transcript" was measured and does not work: on
a Portuguese transcript and on a French one, every single line came back in
English. What works is naming the language, which the app already knows —
Whisper detects it on its way to the words (`transcriber.detected_language`)
and it is handed to the rewrite, which then says "write in Portuguese" rather
than something a model can read as a suggestion. With that, the same two
transcripts came back entirely in their own language. A second, mechanical
check drops any line that still reads as English on a video that was not, so
the failure costs that frame the sentence as spoken rather than a caption in
the wrong language.

`rewrite` needs `captions`, since a rewrite of a transcription nobody asked
for is nothing; set without it, the backend ignores it. Like `captions` and
the frame count, it is read once as the run starts.

And like everything else in this app that loads a model, it is never fatal. A
machine that has not fetched the weights, a generation that comes back
malformed, a line that comes back too long to set — each leaves that frame
with the transcript as spoken. Plainer, not wrong.

The weights are about 3 GB and are fetched once into `backend/models/`, by the
server's own warm-up rather than by the installer — the same arrangement
Whisper has, and for the same reason: a capture channel is not what most runs
use, and an install should not be three gigabytes longer for everyone in order
to save the first capture run a wait.

## The `use` key

Any object may be written as `{"use": "<preset-id>"}` plus overrides, and the
preset from `presets/` is merged underneath it:

```json
"panel": { "use": "title-slab", "padding": 28 }
```

That is there so two channels sharing a design share one definition of it,
rather than holding two copies that drift the first time only one is edited.
