/**
 * Telling a user who has gone elsewhere that their video is ready.
 *
 * A run takes minutes, and the natural thing to do with minutes is another
 * tab. The run carries on without the page in front (the poll keeps going in
 * a background tab, only throttled), but nothing used to say when it was
 * over: the user came back to check, found it still going, left again, and
 * the grid sat finished for however long it took them to look.
 *
 * What a page may NOT do is bring its own tab to the front. Browsers refuse
 * that on purpose — a page that could would be a pop-up that steals the
 * screen — and window.focus() from a background tab is simply ignored. The
 * one door they leave open is a click on the page's own system notification:
 * inside that click the page is allowed to focus itself, and the browser
 * switches to its tab. So the notice is layered:
 *
 *   1. a system notification — Windows draws it over every window, and a
 *      click on it brings this tab forward. Needs the user's permission,
 *      asked for when they press Process, and a secure origin (localhost
 *      counts; the app reached over plain http by address does not);
 *   2. a short chime, for a user looking at another screen entirely;
 *   3. the tab's title blinking, for the one that only glances at the tab
 *      strip, and for any browser where 1 is unavailable or refused;
 *   4. the in-page pop-up, waiting on the page for whenever they come back,
 *      so the news is still there to read after the rest has gone by.
 *
 * Only when the user is actually away (tab hidden, or the window not the one
 * in focus). Someone watching the bar sees the grid appear, and a dialog on
 * top of it then would only be something to close.
 */
import { el } from "./config.js";

const TITLE = document.title;
const BLINK_MS = 1000;

let blinkTimer = null;
let audio = null;

const canNotify = () => "Notification" in window && window.isSecureContext;

/**
 * Called from the click that starts a run, because both things it prepares
 * need that click: browsers only show the permission prompt in answer to a
 * gesture, and only let a page make sound once it has had one. The prompt is
 * asked once — a refusal is the user's answer, and asking again on every
 * video would be nagging. Never awaited: the run must not wait on a prompt
 * the user may leave unanswered.
 */
export function primeReadyNotice() {
    if (canNotify() && Notification.permission === "default") {
        Notification.requestPermission().catch(() => { /* stays "default" */ });
    }
    try {
        audio = audio || new (window.AudioContext || window.webkitAudioContext)();
        audio.resume().catch(() => {});
    } catch { audio = null; /* no sound, the rest still works */ }
}

/**
 * The run has finished and its frames are going onto the grid. `count` is how
 * many, for the wording.
 */
export function announceReady(count) {
    if (!document.hidden && document.hasFocus()) return;
    const what = `${count} ${count === 1 ? "frame is" : "frames are"} waiting in the grid.`;
    showSystemNotice(what);
    chime();
    blinkTitle();
    showDialog(what);
}

function showSystemNotice(body) {
    if (!canNotify() || Notification.permission !== "granted") return;
    try {
        const n = new Notification("Your video has finished processing", {
            body: `${body} Click to open Thumbnail Maker.`,
            icon: "fox.png",
            // Stays on screen until it is dealt with: a toast that slid away
            // while the user was reading something else told them nothing.
            requireInteraction: true,
            // One at a time: a second run finishing replaces the first notice
            // rather than stacking up beside it.
            tag: "video-ready",
        });
        n.onclick = () => {
            // The one moment the browser lets a page come forward (see the
            // top of this file). parent.focus too, for the browsers that
            // only raise the window when asked through it.
            window.focus();
            try { window.parent.focus(); } catch { /* cross-origin parent */ }
            n.close();
        };
        // Coming back by any other route makes the notification old news.
        document.addEventListener("visibilitychange", () => {
            if (!document.hidden) n.close();
        }, { once: true });
    } catch { /* Android Chrome has no page notifications; the rest still runs */ }
}

// Two soft rising notes — enough to be heard across the room, short enough
// not to be an alarm.
function chime() {
    if (!audio || audio.state !== "running") return;
    const start = audio.currentTime;
    [660, 880].forEach((freq, i) => {
        const t = start + i * 0.18;
        const osc = audio.createOscillator();
        const gain = audio.createGain();
        osc.type = "sine";
        osc.frequency.value = freq;
        gain.gain.setValueAtTime(0.0001, t);
        gain.gain.exponentialRampToValueAtTime(0.25, t + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.45);
        osc.connect(gain).connect(audio.destination);
        osc.start(t);
        osc.stop(t + 0.5);
    });
}

// Alternates the tab's title until the user is back on the page. Background
// tabs run timers at most once a second, which is the pace chosen anyway.
function blinkTitle() {
    stopBlink();
    let on = false;
    blinkTimer = setInterval(() => {
        on = !on;
        document.title = on ? "(!) Your video is ready" : TITLE;
    }, BLINK_MS);
}

function stopBlink() {
    if (blinkTimer) clearInterval(blinkTimer);
    blinkTimer = null;
    document.title = TITLE;
}

document.addEventListener("visibilitychange", () => {
    if (!document.hidden) stopBlink();
});
window.addEventListener("focus", stopBlink);

function showDialog(body) {
    el("readyNoticeText").textContent = body;
    el("readyNotice").classList.remove("hidden");
    el("readyNoticeOkBtn").focus();
}

function hideDialog() {
    el("readyNotice").classList.add("hidden");
    stopBlink();
}

el("readyNoticeOkBtn").addEventListener("click", hideDialog);
// A click on the backdrop, outside the card, closes it too: the dialog asks
// nothing, so any way of putting it away is the right answer.
el("readyNotice").addEventListener("click", (e) => {
    if (e.target === e.currentTarget) hideDialog();
});
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !el("readyNotice").classList.contains("hidden")) hideDialog();
});
