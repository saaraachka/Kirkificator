Files for the John Pork call page at `/pork`:

- `john-pork.jpg` (required for the full effect): his face. Used for the caller photo, the blurred background and the video call. Keep it small so the page looks right within a second of the NFC tap: about 800 px on the long side and under ~150 KB (e.g. `ffmpeg -i in.jpg -vf scale=800:-2 -q:v 5 john-pork.jpg`). Without it the page shows a grey "JP" avatar.
- `answer.mp4` (optional): a short, silent clip played on loop when the call is answered. Use H.264 MP4 so iOS Safari plays it, ideally a few seconds and under ~2 MB (e.g. `ffmpeg -i in.mov -an -vf scale=720:-2 -c:v libx264 -profile:v main -crf 28 -movflags +faststart answer.mp4`). Without it the call shows the still photo.
