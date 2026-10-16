# Kirkificator

A Halloween fortune-teller prank. "Madame Sarah" reads your face with the front camera, makes a show of consulting the spirits, and then reveals your destiny: your selfie with someone else's face swapped onto it.

It is a static site with no backend. Face detection (MediaPipe Face Landmarker, loaded as WebAssembly from a CDN), the triangle-mesh warp, colour matching and blending all run in the browser, and no image is ever uploaded anywhere. The only network requests are for the page itself, the MediaPipe library and model, and the Google Fonts.

## Adding or swapping the source face

Put the face you want to swap in at `assets/source-face.jpg`. That path is fixed in `js/faceswap.js` (`SOURCE_FACE_URL`).

Pick a clear, front-facing, evenly lit photo with a neutral or closed-mouth expression and nothing covering the face (no hair over the forehead, no glasses if you can help it). Around 600 to 1200 px on the longest side is plenty; larger images are downscaled to 1024 px. If the face can't be detected the app shows a "crystal ball is clouded" error, so you'll know straight away.

The landmarks for the source face are computed once when the page loads, so swapping the image is just replacing the file and redeploying.

## Tuning

At the top of `js/faceswap.js`:

- `BLEND_STRENGTH` (0 to 1) controls how strongly the source face replaces the selfie. 1 is a full swap, around 0.6 to 0.7 gives a subtler "something's off" look.
- `FEATHER` is the width of the soft edge around the face, as a fraction of face width.
- `MIN_CONTRAST_GAIN` / `MAX_CONTRAST_GAIN` limit how far colour matching can stretch contrast.

The loading lines and their timing live at the top of `js/app.js` (`PROCESSING_LINES`, `LINE_MS`).

## Running locally

Camera access needs a secure context. `localhost` counts, so any static server works for desktop testing:

```sh
python3 -m http.server 8000
# open http://localhost:8000
```

To try it on a phone you need HTTPS. The easiest options are deploying (below) or a tunnel such as `npx localtunnel --port 8000` or `cloudflared tunnel --url http://localhost:8000`.

## Deploying

The site is plain files in the repo root (`index.html`, `css/`, `js/`, `assets/`), with no build step. Both GitHub Pages and Netlify serve over HTTPS, which the camera requires.

**GitHub Pages:** push to GitHub, then go to Settings → Pages, set the source to "Deploy from a branch", pick your branch and the `/ (root)` folder, and save. The site appears at `https://<user>.github.io/<repo>/` within a minute or two. All paths in the app are relative, so it works from a sub-path.

**Netlify:** either drag the project folder onto https://app.netlify.com/drop, or connect the repo with the build command left empty and the publish directory set to the repo root.

## How it works

- `js/app.js` drives the screens: intro → camera with a 3-2-1 countdown → processing lines → result (or a themed error). It also bakes the "YOUR DESTINY" stamp into the image and handles Save/Share via the Web Share API, falling back to a download.
- `js/camera.js` opens the front camera (`facingMode: 'user'`) and captures exactly the mirrored, cropped frame shown in the preview, capped at 960 px.
- `js/faceswap.js` detects 468 face landmarks on both images, Delaunay-triangulates the source landmarks once (`js/delaunay.js`), and draws each source triangle onto the matching selfie triangle with WebGL (falling back to Canvas 2D affine transforms without WebGL). It then matches the warped face's per-channel mean and standard deviation to the selfie's skin inside the face oval and blends it in through a feathered face-oval mask.

Tested in Chromium; it targets iOS Safari 15+ and current Android Chrome. On iOS the Web Share sheet offers "Save Image"; on browsers without file sharing the image downloads instead, and press-and-hold on the result image also works.
