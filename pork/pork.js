// John Pork is calling. Incoming -> (accept) video call -> (end) call ended -> he calls back;
// incoming -> (decline) black -> he calls back. No camera, microphone, sound or vibration.

(() => {
  const VIDEO_URL = '../assets/pork/answer.mp4';
  const FREEZE_AFTER_MS = 6000; // the video call freezes and the banner drops in
  const DECLINE_BLACK_MS = 1500; // screen stays black after a decline before he calls back
  const ENDED_MS = 1500; // "Call Ended" is shown this long
  const CALLBACK_GAP_MS = 2000; // then black this long before he calls back

  const $ = (id) => document.getElementById(id);
  const body = document.body;
  const callScreen = $('call');
  const video = $('feed-video');

  let attempt = 1; // which call this is: 1 = first, 2 = "(2)", ... 4+ = "(urgent)"
  let videoMissing = false; // set once the optional clip fails to load
  let callStart = 0;
  let timer = 0;
  let freezeTimer = 0;
  let busy = false; // ignore taps during transitions

  const callerName = (n) => (n <= 1 ? 'John Pork' : n <= 3 ? `John Pork (${n})` : 'John Pork (urgent)');
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const fmt = (ms) => {
    const s = Math.floor(ms / 1000);
    return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  };

  function show(screen) {
    body.dataset.screen = screen;
  }

  function incoming() {
    $('caller-name').textContent = callerName(attempt);
    show('incoming');
    body.classList.remove('dark');
    busy = false;
  }

  async function decline() {
    if (busy) return;
    busy = true;
    body.classList.add('dark');
    await wait(DECLINE_BLACK_MS);
    attempt++;
    incoming();
  }

  function accept() {
    if (busy) return;
    busy = true;
    callScreen.classList.remove('frozen', 'has-video');
    $('banner').classList.remove('show');
    // The still photo shows until the clip is actually playing (iOS won't preload video
    // without a tap, so this is the first chance to start it).
    if (!videoMissing && video.src) {
      try { video.currentTime = 0; } catch { /* not loaded yet */ }
      video.play().catch(() => {});
    }
    callStart = Date.now();
    $('call-timer').textContent = '00:00';
    timer = setInterval(() => { $('call-timer').textContent = fmt(Date.now() - callStart); }, 250);
    show('call');
    freezeTimer = setTimeout(freeze, FREEZE_AFTER_MS);
    busy = false;
  }

  function freeze() {
    callScreen.classList.add('frozen');
    video.pause();
    $('banner').classList.add('show');
  }

  async function end() {
    if (busy || body.dataset.screen !== 'call') return;
    busy = true;
    clearTimeout(freezeTimer);
    clearInterval(timer);
    video.pause();
    $('ended-time').textContent = fmt(Date.now() - callStart);
    show('ended');
    await wait(ENDED_MS);
    body.classList.add('dark');
    await wait(CALLBACK_GAP_MS);
    attempt++;
    incoming();
  }

  // The answered-call video is optional: load it quietly once the page is up, and fall
  // back to the still photo if it's missing or won't play.
  function loadVideo() {
    video.addEventListener('playing', () => {
      if (body.dataset.screen === 'call' && !callScreen.classList.contains('frozen')) callScreen.classList.add('has-video');
    });
    video.addEventListener('error', () => {
      videoMissing = true;
      callScreen.classList.remove('has-video');
    });
    video.preload = 'auto';
    video.src = VIDEO_URL;
    video.load();
  }

  $('btn-accept').addEventListener('click', accept);
  $('btn-decline').addEventListener('click', decline);
  $('btn-end').addEventListener('click', end);

  // No scrolling, pinch zoom or double-tap zoom (iOS ignores user-scalable=no).
  const block = (e) => e.preventDefault();
  document.addEventListener('touchmove', block, { passive: false });
  document.addEventListener('gesturestart', block);
  document.addEventListener('dblclick', block);
  document.addEventListener('contextmenu', block);

  if (document.readyState === 'complete') loadVideo();
  else window.addEventListener('load', loadVideo, { once: true });
})();
