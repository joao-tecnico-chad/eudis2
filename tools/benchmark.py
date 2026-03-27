"""Benchmark OAK-1W performance — find the bottleneck.

Tests different configurations to measure:
  1. Pure NN inference FPS (no video encoder)
  2. NN + MJPEG encoder FPS
  3. Effect of inference threads (1 vs 2)
  4. USB speed

Usage:
    python tools/benchmark.py --model models/yolov5s.rvc2.tar.xz
    python tools/benchmark.py --model models/yolov6s_416.rvc2.tar.xz
"""

import argparse
import time

import depthai as dai

parser = argparse.ArgumentParser(description="OAK benchmark")
parser.add_argument("--model", required=True, help="Path to NNArchive")
parser.add_argument("--duration", type=int, default=15, help="Seconds per test")
args = parser.parse_args()


def run_test(name, setup_fn):
    """Run a benchmark test and return FPS."""
    print(f"\n{'='*50}")
    print(f"TEST: {name}")
    print(f"{'='*50}")

    with dai.Pipeline() as p:
        queues = setup_fn(p)
        p.start()

        # Warmup
        time.sleep(2)

        # Measure
        count = 0
        start = time.monotonic()
        while time.monotonic() - start < args.duration:
            for q in queues:
                msg = q.tryGet()
                if msg is not None:
                    count += 1
            time.sleep(0.0005)

        elapsed = time.monotonic() - start
        fps = count / elapsed
        print(f"  Result: {fps:.1f} FPS ({count} frames in {elapsed:.1f}s)")
        return fps


def test_nn_only_1thread(p):
    """Pure NN inference, 1 thread, no video encoder."""
    cam = p.create(dai.node.Camera).build()
    nn_archive = dai.NNArchive(args.model)
    nn_size = nn_archive.getInputSize()
    cam_out = cam.requestOutput((nn_size[0], nn_size[1]), dai.ImgFrame.Type.BGR888p)
    det = p.create(dai.node.DetectionNetwork).build(cam_out, nn_archive)
    det.setConfidenceThreshold(0.5)
    det.setNumInferenceThreads(1)
    q = det.out.createOutputQueue(maxSize=1, blocking=False)
    return [q]


def test_nn_only_2thread(p):
    """Pure NN inference, 2 threads, no video encoder."""
    cam = p.create(dai.node.Camera).build()
    nn_archive = dai.NNArchive(args.model)
    nn_size = nn_archive.getInputSize()
    cam_out = cam.requestOutput((nn_size[0], nn_size[1]), dai.ImgFrame.Type.BGR888p)
    det = p.create(dai.node.DetectionNetwork).build(cam_out, nn_archive)
    det.setConfidenceThreshold(0.5)
    det.setNumInferenceThreads(2)
    q = det.out.createOutputQueue(maxSize=1, blocking=False)
    return [q]


def test_nn_plus_mjpeg(p):
    """NN + hardware MJPEG encoder (same as detect.py)."""
    cam = p.create(dai.node.Camera).build()
    nn_archive = dai.NNArchive(args.model)
    nn_size = nn_archive.getInputSize()
    cam_out = cam.requestOutput((nn_size[0], nn_size[1]), dai.ImgFrame.Type.BGR888p)
    det = p.create(dai.node.DetectionNetwork).build(cam_out, nn_archive)
    det.setConfidenceThreshold(0.5)
    det.setNumInferenceThreads(2)

    enc_out = cam.requestOutput((640, 480), dai.ImgFrame.Type.NV12)
    encoder = p.create(dai.node.VideoEncoder)
    encoder.setDefaultProfilePreset(15, dai.VideoEncoderProperties.Profile.MJPEG)
    encoder.setQuality(70)
    enc_out.link(encoder.input)

    q_det = det.out.createOutputQueue(maxSize=1, blocking=False)
    q_mjpeg = encoder.out.createOutputQueue(maxSize=1, blocking=False)
    return [q_det]


def test_nn_plus_mjpeg_small(p):
    """NN + smaller MJPEG (320x240) to reduce encoder load."""
    cam = p.create(dai.node.Camera).build()
    nn_archive = dai.NNArchive(args.model)
    nn_size = nn_archive.getInputSize()
    cam_out = cam.requestOutput((nn_size[0], nn_size[1]), dai.ImgFrame.Type.BGR888p)
    det = p.create(dai.node.DetectionNetwork).build(cam_out, nn_archive)
    det.setConfidenceThreshold(0.5)
    det.setNumInferenceThreads(2)

    enc_out = cam.requestOutput((320, 240), dai.ImgFrame.Type.NV12)
    encoder = p.create(dai.node.VideoEncoder)
    encoder.setDefaultProfilePreset(10, dai.VideoEncoderProperties.Profile.MJPEG)
    encoder.setQuality(50)
    enc_out.link(encoder.input)

    q_det = det.out.createOutputQueue(maxSize=1, blocking=False)
    q_mjpeg = encoder.out.createOutputQueue(maxSize=1, blocking=False)
    return [q_det]


def test_nn_max_shaves(p):
    """NN with explicit max shaves (6 per thread)."""
    cam = p.create(dai.node.Camera).build()
    nn_archive = dai.NNArchive(args.model)
    nn_size = nn_archive.getInputSize()
    cam_out = cam.requestOutput((nn_size[0], nn_size[1]), dai.ImgFrame.Type.BGR888p)
    det = p.create(dai.node.DetectionNetwork).build(cam_out, nn_archive)
    det.setConfidenceThreshold(0.5)
    det.setNumInferenceThreads(2)
    det.setNumShavesPerInferenceThread(6)
    q = det.out.createOutputQueue(maxSize=1, blocking=False)
    return [q]


def test_nn_1thread_max_shaves(p):
    """NN 1 thread with all shaves (13)."""
    cam = p.create(dai.node.Camera).build()
    nn_archive = dai.NNArchive(args.model)
    nn_size = nn_archive.getInputSize()
    cam_out = cam.requestOutput((nn_size[0], nn_size[1]), dai.ImgFrame.Type.BGR888p)
    det = p.create(dai.node.DetectionNetwork).build(cam_out, nn_archive)
    det.setConfidenceThreshold(0.5)
    det.setNumInferenceThreads(1)
    det.setNumShavesPerInferenceThread(13)
    q = det.out.createOutputQueue(maxSize=1, blocking=False)
    return [q]


def test_nn_2thread_6shaves_nce(p):
    """NN 2 threads, 6 shaves, 2 NCEs."""
    cam = p.create(dai.node.Camera).build()
    nn_archive = dai.NNArchive(args.model)
    nn_size = nn_archive.getInputSize()
    cam_out = cam.requestOutput((nn_size[0], nn_size[1]), dai.ImgFrame.Type.BGR888p)
    det = p.create(dai.node.DetectionNetwork).build(cam_out, nn_archive)
    det.setConfidenceThreshold(0.5)
    det.setNumInferenceThreads(2)
    det.setNumShavesPerInferenceThread(6)
    det.setNumNCEPerInferenceThread(2)
    q = det.out.createOutputQueue(maxSize=1, blocking=False)
    return [q]


# Check USB speed
device = dai.Device()
usb_speed = device.getUsbSpeed()
print(f"USB Speed: {usb_speed.name}")
device.close()

# Run benchmarks
results = {}
results["NN only (1 thread)"] = run_test("NN only (1 thread)", test_nn_only_1thread)
results["NN only (2 threads)"] = run_test("NN only (2 threads)", test_nn_only_2thread)
results["NN + MJPEG 640x480"] = run_test("NN + MJPEG 640x480", test_nn_plus_mjpeg)
results["NN + MJPEG 320x240"] = run_test("NN + MJPEG 320x240", test_nn_plus_mjpeg_small)
results["NN 2t x 6 shaves"] = run_test("NN 2 threads x 6 shaves", test_nn_max_shaves)
results["NN 1t x 13 shaves"] = run_test("NN 1 thread x 13 shaves", test_nn_1thread_max_shaves)
results["NN 2t x 6sh + 2NCE"] = run_test("NN 2 threads x 6 shaves + 2 NCE", test_nn_2thread_6shaves_nce)

print(f"\n{'='*50}")
print(f"SUMMARY — {args.model}")
print(f"{'='*50}")
for name, fps in results.items():
    bar = "█" * int(fps)
    print(f"  {name:25s} {fps:5.1f} FPS  {bar}")
