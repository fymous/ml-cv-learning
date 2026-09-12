# Test videos

Put public / your-own sample clips here. Video files are gitignored.

Suggested starting clip: any short indoor walking video (door + people).

Then:

```bash
cd worker
python -m app.main --source ../testdata/your-clip.mp4 --zones ../testdata/zones.yaml
```

Edit `zones.yaml` so the polygons match the clip. Coordinates are 0–1.
