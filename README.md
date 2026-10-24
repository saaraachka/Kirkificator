# Kirkificator

A carnival sideshow prank. "Step Right Up" congratulates you on being selected for the main exhibit, takes your photo with the front camera, turns your face to clay and kneads it while the swap runs, and then the new face pushes up through the clay and hardens into the new attraction: your selfie with someone else's face swapped onto it, stamped YOU'VE BEEN KIRKED.

It is a static site plus two Netlify Functions. The face swap itself is done by an AI face-swap model on [Replicate](https://replicate.com), called through the functions so the API token stays on the server. The browser first checks for a face with MediaPipe Face Landmarker (loaded as WebAssembly from a CDN), so photos without a face are never uploaded. If the AI swap fails or takes longer than 45 seconds, the app falls back to the original on-device swap (triangle-mesh warp, colour matching and blending in the browser), so a reading always finishes.

## Setup on Netlify

1. Connect this GitHub repo to a Netlify site. `netlify.toml` already sets everything: no build command, the repo root as the publish directory, and `netlify/functions` as the functions directory.
2. In Netlify, under **Site configuration → Environment variables**, add:
   - `REPLICATE_API_TOKEN` (required): your token from https://replicate.com/account/api-tokens. Never commit it to the repo.
   - `REPLICATE_MODEL` (optional): which model to run, as `owner/name:version` or `owner/name`. It defaults to [`cdingram/face-swap`](https://replicate.com/cdingram/face-swap), pinned to a version. Another model only works as a drop-in if it takes the same inputs (`input_image` for the selfie, `swap_image` for the face) and returns one image URL. Otherwise change the input names in `netlify/functions/swap-start.js`.
   Redeploy after changing environment variables.
3. **Set a spending limit on Replicate** (Account → Billing). The functions are public endpoints, so anyone who finds the site can start swaps on your account.
4. Write the site's URL to the NFC tag, e.g. `https://<your-site>.netlify.app/`. The camera needs HTTPS, which Netlify provides.

GitHub Pages and other static hosts can't run the functions, so there the app always uses the on-device swap.

Replicate downloads the source face from the deployed site at `/assets/source-face.jpg`, so that file has to be publicly reachable at the site's own address.

Files: `netlify/functions/swap-start.js` validates the upload (JPEG or PNG, 4 MB at most) and starts a prediction, returning only its id. `netlify/functions/swap-status.js` returns only `{ status, outputUrl }` for a prediction. Netlify cuts sync functions off at about 10 seconds, so neither one waits for the model: the browser polls `swap-status` every 1.5 s.

## Adding or swapping the source face

Put the face you want to swap in at `assets/source-face.jpg`. That path is fixed in `js/faceswap.js` (`SOURCE_FACE_URL`, for the on-device fallback) and `netlify/functions/swap-start.js` (`SOURCE_FACE_PATH`, which Replicate downloads).

Pick a clear, front-facing, evenly lit photo with a neutral or closed-mouth expression and nothing covering the face (no hair over the forehead, no glasses if you can help it). Around 600 to 1200 px on the longest side is plenty; larger images are downscaled to 1024 px. If the face can't be detected the on-device fallback shows an "exhibit's star has no face" error, so you'll know straight away.

The landmarks for the source face are computed once when the page loads, so swapping the image is just replacing the file and redeploying.

## Tuning

The AI swap's settings are at the top of `js/faceswap.js`: `UPLOAD_MAX_SIZE` (1024 px) and `UPLOAD_QUALITY` (0.85) for the uploaded selfie, `POLL_MS` (1.5 s) and `REMOTE_TIMEOUT_MS` (45 s) for polling. A selfie that encodes larger than `UPLOAD_TARGET_BYTES` (256 KB) is re-encoded at lower quality, because Replicate recommends inline data URLs only for small files.

The on-device fallback is tuned with these, also at the top of `js/faceswap.js`:

- `BLEND_STRENGTH` (0 to 1, default 1) controls how strongly the source face replaces the selfie. 1 is a full paste, around 0.6 to 0.7 gives a subtler "something's off" look.
- `SHAPE_BLEND` (0 to 1, default 0.35) controls how much the source face bends to fit the selfie. At 0 it is only rotated, scaled and moved onto the head, so it keeps its own proportions (the "kirkified" look). At 1 every landmark is pulled onto the selfie's, which ends up looking like the selfie person.
- `LOWFREQ_RADIUS` (default 0.06, as a fraction of face width) is the blur radius that splits the face into broad colour and lighting (taken from the selfie) and detail (taken from the source). Larger keeps more of the source's own shading; smaller takes more of the selfie's lighting.
- `DETAIL_STRENGTH` (default 1) scales the source's features and texture laid over the selfie's skin.
- `BROW_DARKEN` (default 1) scales that detail inside the eyebrows only. Raise it if the brows look faint, lower it if they look too heavy.
- `FOREHEAD_EXTENSION` (default 0.1) is how far the pasted region reaches above the face oval, as a fraction of face height.
- `TOP_FADE` (default 0.2) is the distance, as a fraction of face height, over which the top of the face fades out. It reaches zero at the top-of-forehead landmark (the source's or the selfie's, whichever is lower), so neither person's hairline is pasted over.
- `FEATHER` (default 0.08) is the width of the soft edge along the cheeks and jaw, as a fraction of face width.
- `PYRAMID_LEVELS` (default 6) is the maximum number of levels used for multi-band blending. More levels spread colour and lighting differences over a wider area.

The clayface loading screen is tuned at the top of `js/clay.js`:

- `CLAY_IN_MS` (1.2 s): how long the face takes to turn to clay.
- `MIN_KNEAD_MS` (4 s): the least kneading shown, however fast the swap is.
- `BECOME_MS` (2.6 s): how long the new face takes to push up through the clay.
- `HARDEN_MS` (1.2 s): how long the clay takes to settle and fade into the photo.
- `MOLD_INTENSITY` (1): how hard the invisible hands mold the face. It drops to about a third for people who have reduced motion turned on.
- `DRIP_AMOUNT` (1): how far the jaw drips when it sags.
- `CLAY_TEXTURE_STRENGTH` (1): fingerprints, thumb smears and surface grain.

The lines shown over the kneading are `KNEAD_LINES` at the top of `js/app.js`. They loop for as long as the swap takes, each fully visible for `LINE_MS` (2.6 s) with a `FADE_MS` (0.9 s) fade.

The theme lives in `css/styles.css`: colour tokens at the top, with paper texture and film grain made from CSS gradients and inline SVG noise (no image files).

## Running locally

The functions need the Netlify CLI. Camera access needs a secure context, and `localhost` counts:

```sh
npx netlify-cli dev
# open http://localhost:8888
```

`netlify dev` reads `REPLICATE_API_TOKEN` from a local `.env` file (ignored by git) or from the linked Netlify site. Locally the AI swap can't succeed anyway: Replicate has to download the source face from the site's own URL, and it can't reach `localhost`. You'll get the on-device fallback, which is also the quickest way to test that path. To test the AI path, deploy (a Netlify deploy preview works).

To try it on a phone you need HTTPS. The easiest option is deploying, or a tunnel such as `cloudflared tunnel --url http://localhost:8888`.

## How it works

- `js/app.js` drives the screens: intro → camera → clayface → result (or an error). The camera screen runs MediaPipe on the live preview a few times a second, and the oval guide glows gold and "TAKE IT" unlocks only while a face is found. Tapping it captures instantly with a flash, and the frozen shot waits for "USE IT" or "RETAKE", so nothing is uploaded until the user approves it. After "USE IT" it finds the face landmarks, starts the clayface, then tries the AI swap and falls back to the on-device swap if that fails or times out. It also bakes the "YOU'VE BEEN KIRKED" ticket stamp into the image (two lines, red ink on a faint cream backing so it reads over any clothes). It handles sharing via the Web Share API, falling back to a download: "SHOW THE OTHERS" shares the still, and "SHOW THE TRANSFORMATION" shares the recorded clip. That button is hidden where the browser can't record.
- `js/clay.js` is the loading screen. It renders full-screen in WebGL, with a simpler Canvas 2D fallback (blur, posterize and emboss, a strip-warp wobble and a crossfade). Clay styling is precomputed once per image on the GPU: a bilateral smoothing pass, light posterizing, and a height map from blurred luminance. The surface relief (thumb smears, partial fingerprints, grain) is baked once too. Each frame then renders the kneading into a small displacement texture, shades the clay with normals from the height map plus the thumb dents (matte with a soft waxy highlight), and vignettes the frame. The kneading is a random, overlapping sequence of landmark-driven moves (cheeks pulled out, nose pushed in and pulled out, eyes squeezed, mouth stretched, forehead pressed flat, jaw sagging and dripping, twists, squashes), plus thumb dents that fill back in and a constant wobble. Becoming is a mesh warp from the selfie's landmarks to the result's, with a crossfade between the two clay faces. The canvas is recorded with MediaRecorder (MP4 where supported, else WebM) from 2 s before Becoming to the stamped result. When the result arrives after the minimum kneading, Becoming starts 2 s later, so that pre-roll is always there.
- `js/camera.js` opens the front camera (`facingMode: 'user'`) and captures exactly the mirrored, cropped frame shown in the preview, capped at 960 px.
- `js/faceswap.js` has the AI path (`swapFaceRemote`: downscale and upload, poll, then download the result into a canvas so it can be stamped and shared) and the face check (`hasFace`). The on-device fallback (`swapFace`) detects 468 face landmarks on both images, fits a single similarity transform (rotation, uniform scale, translation) from the source's eye centres, forehead and chin to the selfie's, and moves each source landmark there, nudged `SHAPE_BLEND` of the way toward the selfie landmark. An extra ring of points above and around the face oval carries the source's forehead and hairline along. The source points are Delaunay-triangulated once (`js/delaunay.js`) and each triangle is drawn onto its target with WebGL (falling back to Canvas 2D affine transforms without WebGL). The new face is built by frequency separation in Lab colour space (`js/blend.js`): the selfie's skin blurred by `LOWFREQ_RADIUS`, plus the source minus its own blurred skin. Both blurs are masked, normalised blurs of the face skin: a feature mask built from the landmarks (brows, eyes, upper lids and lips, slightly dilated and feathered) is left out of both, and skin colour is filled in smoothly underneath. That way the source's features keep their full darkness and colour in the detail layer, and the selfie's own features can't ghost through, so the result takes the selfie's skin tone, lighting and shading while the features come only from the source. Inside the eye openings the source's own colours are kept so the whites stay white on any skin tone, and the brows take their tint from the source's brow hair. The result is then combined with the selfie by multi-band blending. Both images are split into Laplacian pyramids and each band is blended through a matching Gaussian pyramid of the mask, so colour and lighting blend over a wide area while fine detail keeps a narrow edge. The mask follows the face oval, is eroded and blurred so it tapers to zero at the outline, and fades out across the forehead.

Tested in Chromium; it targets iOS Safari 15+ and current Android Chrome. On iOS the Web Share sheet offers "Save Image"; on browsers without file sharing the image downloads instead, and press-and-hold on the result image also works.
