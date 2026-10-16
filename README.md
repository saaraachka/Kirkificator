# Kirkificator

A Halloween fortune-teller prank. "Madame Sarah" reads your face with the front camera, makes a show of consulting the spirits, and then reveals your destiny: your selfie with someone else's face swapped onto it.

It is a static site with no backend. Face detection (MediaPipe Face Landmarker, loaded as WebAssembly from a CDN), the triangle-mesh warp, colour matching and blending all run in the browser, and no image is ever uploaded anywhere. The only network requests are for the page itself, the MediaPipe library and model, and the Google Fonts.

## Adding or swapping the source face

Put the face you want to swap in at `assets/source-face.jpg`. That path is fixed in `js/faceswap.js` (`SOURCE_FACE_URL`).

Pick a clear, front-facing, evenly lit photo with a neutral or closed-mouth expression and nothing covering the face (no hair over the forehead, no glasses if you can help it). Around 600 to 1200 px on the longest side is plenty; larger images are downscaled to 1024 px. If the face can't be detected the app shows a "crystal ball is clouded" error, so you'll know straight away.

The landmarks for the source face are computed once when the page loads, so swapping the image is just replacing the file and redeploying.

## Tuning

At the top of `js/faceswap.js`:

- `BLEND_STRENGTH` (0 to 1, default 1) controls how strongly the source face replaces the selfie. 1 is a full paste, around 0.6 to 0.7 gives a subtler "something's off" look.
- `SHAPE_BLEND` (0 to 1, default 0.35) controls how much the source face bends to fit the selfie. At 0 it is only rotated, scaled and moved onto the head, so it keeps its own proportions (the "kirkified" look). At 1 every landmark is pulled onto the selfie's, which ends up looking like the selfie person.
- `LOWFREQ_RADIUS` (default 0.06, as a fraction of face width) is the blur radius that splits the face into broad colour and lighting (taken from the selfie) and detail (taken from the source). Larger keeps more of the source's own shading; smaller takes more of the selfie's lighting.
- `DETAIL_STRENGTH` (default 1) scales the source's features and texture laid over the selfie's skin.
- `FOREHEAD_EXTENSION` (default 0.1) is how far the pasted region reaches above the face oval, as a fraction of face height.
- `TOP_FADE` (default 0.2) is the distance, as a fraction of face height, over which the top of the face fades out. It reaches zero at the top-of-forehead landmark (the source's or the selfie's, whichever is lower), so neither person's hairline is pasted over.
- `FEATHER` (default 0.08) is the width of the soft edge along the cheeks and jaw, as a fraction of face width.
- `PYRAMID_LEVELS` (default 6) is the maximum number of levels used for multi-band blending. More levels spread colour and lighting differences over a wider area.

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
- `js/faceswap.js` detects 468 face landmarks on both images, fits a single similarity transform (rotation, uniform scale, translation) from the source's eye centres, forehead and chin to the selfie's, and moves each source landmark there, nudged `SHAPE_BLEND` of the way toward the selfie landmark. An extra ring of points above and around the face oval carries the source's forehead and hairline along. The source points are Delaunay-triangulated once (`js/delaunay.js`) and each triangle is drawn onto its target with WebGL (falling back to Canvas 2D affine transforms without WebGL). The new face is built by frequency separation in Lab colour space (`js/blend.js`): the selfie's skin blurred by `LOWFREQ_RADIUS`, plus the source minus its own blurred skin. Both blurs are skin-only, with the eyes, brows and lips cut out and filled from the surrounding skin, so the result takes the selfie's skin tone, lighting and shading while the features come only from the source. The eyes and brows keep the source's own colours, shifted only part way toward the selfie's lighting, so the whites stay white and the brows keep their hair colour on any skin tone. The result is then combined with the selfie by multi-band blending. Both images are split into Laplacian pyramids and each band is blended through a matching Gaussian pyramid of the mask, so colour and lighting blend over a wide area while fine detail keeps a narrow edge. The mask follows the face oval, is eroded and blurred so it tapers to zero at the outline, and fades out across the forehead.

Tested in Chromium; it targets iOS Safari 15+ and current Android Chrome. On iOS the Web Share sheet offers "Save Image"; on browsers without file sharing the image downloads instead, and press-and-hold on the result image also works.
