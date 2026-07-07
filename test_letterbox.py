"""
Unit test for LetterboxCompositionDataset — letterbox scaling + box coordinate correction.

Tests with synthetic random images so it needs no real data.
"""
import os
import json
import tempfile
import unittest

import torch
import numpy as np
from PIL import Image

from train_photography_eye import LetterboxCompositionDataset

TARGET = 224


def _make_image(w, h, seed=0):
    """Create a random RGB image of the given size."""
    rng = np.random.RandomState(seed)
    arr = (rng.rand(h, w, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _make_labelme_json(left, top, right, bottom):
    """Return a LabelMe-format dict with a single rectangle."""
    return {
        "shapes": [
            {
                "label": "composition",
                "shape_type": "rectangle",
                "points": [[left, top], [right, bottom]],
            }
        ]
    }


class TestLetterboxCompositionDataset(unittest.TestCase):
    """Tests for LetterboxCompositionDataset."""

    def _make_dataset(self, images_with_boxes, target_size=TARGET):
        """
        Write image+JSON pairs into a temp dir and return (dataset, tmpdir).

        images_with_boxes: list of (img: PIL.Image, box_xyxy: [x1,y1,x2,y2] in raw-pixel coords)
        """
        tmpdir = tempfile.TemporaryDirectory()
        for idx, (img, box) in enumerate(images_with_boxes):
            name = f"{idx}.jpg"
            img.save(os.path.join(tmpdir.name, name))
            with open(os.path.join(tmpdir.name, f"{idx}.json"), "w") as f:
                json.dump(_make_labelme_json(*box), f)

        ds = LetterboxCompositionDataset(tmpdir.name, target_size=target_size)
        return ds, tmpdir

    # ==============================================================
    # 1. 输出尺寸 & 张量类型
    # ==============================================================

    def test_output_size_and_dtype(self):
        """Every sample must be (3, 224, 224) float32 tensor."""
        images = [
            (_make_image(400, 300, 0), [50, 40, 350, 260]),   # landscape
            (_make_image(200, 400, 1), [20, 30, 180, 370]),   # portrait
            (_make_image(224, 224, 2), [10, 10, 214, 214]),   # already square
        ]
        ds, tmpdir = self._make_dataset(images)
        self.assertEqual(len(ds), 3)

        for i in range(3):
            img_t, box_t = ds[i]
            self.assertIsInstance(img_t, torch.Tensor, f"Sample {i}: expected Tensor")
            self.assertEqual(
                img_t.shape, (3, TARGET, TARGET),
                f"Sample {i}: shape {img_t.shape} != (3, {TARGET}, {TARGET})"
            )
            self.assertEqual(img_t.dtype, torch.float32, f"Sample {i}: dtype mismatch")
            self.assertEqual(box_t.dtype, torch.float32)

        tmpdir.cleanup()

    # ==============================================================
    # 2. Letterbox 等比缩放 & 黑边正确性
    # ==============================================================

    def _letterbox_checks(self, ds, idx, orig_w, orig_h):
        """Verify letterbox for sample idx — expected scaler & padding."""
        img_t, _ = ds[idx]

        scale = TARGET / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_x = (TARGET - new_w) // 2
        pad_y = (TARGET - new_h) // 2

        img_np = img_t.permute(1, 2, 0).numpy()  # (H,W,C)

        # --- 上下黑边 ---
        if pad_y > 0:
            top_strip = img_np[:pad_y, :, :]
            self.assertAlmostEqual(
                np.max(top_strip), 0.0, delta=1/255,
                msg=f"Sample {idx}: top padding not black"
            )
            bot_strip = img_np[pad_y + new_h:, :, :]
            self.assertAlmostEqual(
                np.max(bot_strip), 0.0, delta=1/255,
                msg=f"Sample {idx}: bottom padding not black"
            )

        # --- 左右黑边 ---
        if pad_x > 0:
            left_strip = img_np[:, :pad_x, :]
            self.assertAlmostEqual(
                np.max(left_strip), 0.0, delta=1/255,
                msg=f"Sample {idx}: left padding not black"
            )
            right_strip = img_np[:, pad_x + new_w:, :]
            self.assertAlmostEqual(
                np.max(right_strip), 0.0, delta=1/255,
                msg=f"Sample {idx}: right padding not black"
            )

        # --- 内容区域非全黑 (确认图片确实贴上了) ---
        content = img_np[pad_y:pad_y + new_h, pad_x:pad_x + new_w, :]
        self.assertGreater(
            np.max(content), 0.01,
            msg=f"Sample {idx}: content area is black — image not pasted"
        )

        return pad_x, pad_y, new_w, new_h

    def test_letterbox_landscape(self):
        """Landscape 400×300 → 224×168 inside, padded top/bottom 28px each."""
        img = _make_image(400, 300, 0)
        ds, tmpdir = self._make_dataset([(img, [50, 40, 350, 260])])
        pad_x, pad_y, new_w, new_h = self._letterbox_checks(ds, 0, 400, 300)
        self.assertEqual(new_w, TARGET)
        self.assertEqual(new_h, 168)
        self.assertEqual(pad_x, 0)
        self.assertEqual(pad_y, 28)
        tmpdir.cleanup()

    def test_letterbox_portrait(self):
        """Portrait 200×400 → 112×224 inside, padded left/right 56px each."""
        img = _make_image(200, 400, 1)
        ds, tmpdir = self._make_dataset([(img, [20, 30, 180, 370])])
        pad_x, pad_y, new_w, new_h = self._letterbox_checks(ds, 0, 200, 400)
        self.assertEqual(new_w, 112)
        self.assertEqual(new_h, TARGET)
        self.assertEqual(pad_x, 56)
        self.assertEqual(pad_y, 0)
        tmpdir.cleanup()

    def test_letterbox_square(self):
        """Square 224×224 → no padding."""
        img = _make_image(224, 224, 2)
        ds, tmpdir = self._make_dataset([(img, [10, 10, 214, 214])])
        pad_x, pad_y, new_w, new_h = self._letterbox_checks(ds, 0, 224, 224)
        self.assertEqual(new_w, TARGET)
        self.assertEqual(new_h, TARGET)
        self.assertEqual(pad_x, 0)
        self.assertEqual(pad_y, 0)
        tmpdir.cleanup()

    # ==============================================================
    # 3. 坐标变换正确性
    # ==============================================================

    def _check_box(self, ds, idx, orig_w, orig_h, raw_box_xyxy, atol=1/224 + 0.005):
        """
        raw_box_xyxy: [x1, y1, x2, y2] in original-pixel coords.
        Verifies returned box_t = correctly transformed normalized box.
        """
        _, box_t = ds[idx]
        x1n, y1n, x2n, y2n = box_t.tolist()

        scale = TARGET / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_x = (TARGET - new_w) // 2
        pad_y = (TARGET - new_h) // 2

        rx1, ry1, rx2, ry2 = raw_box_xyxy
        ex1 = (rx1 * scale + pad_x) / TARGET
        ey1 = (ry1 * scale + pad_y) / TARGET
        ex2 = (rx2 * scale + pad_x) / TARGET
        ey2 = (ry2 * scale + pad_y) / TARGET

        self.assertAlmostEqual(x1n, ex1, delta=atol, msg=f"x1: got {x1n:.4f}, expected {ex1:.4f}")
        self.assertAlmostEqual(y1n, ey1, delta=atol, msg=f"y1: got {y1n:.4f}, expected {ey1:.4f}")
        self.assertAlmostEqual(x2n, ex2, delta=atol, msg=f"x2: got {x2n:.4f}, expected {ex2:.4f}")
        self.assertAlmostEqual(y2n, ey2, delta=atol, msg=f"y2: got {y2n:.4f}, expected {ey2:.4f}")

        # All in [0, 1]
        for val in [x1n, y1n, x2n, y2n]:
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_box_coords_landscape(self):
        """Landscape: centered box, verify transform."""
        img = _make_image(400, 300, 0)
        ds, tmpdir = self._make_dataset([(img, [100, 75, 300, 225])])
        self._check_box(ds, 0, 400, 300, [100, 75, 300, 225])
        tmpdir.cleanup()

    def test_box_coords_portrait(self):
        """Portrait: centered box, verify transform."""
        img = _make_image(200, 400, 1)
        ds, tmpdir = self._make_dataset([(img, [40, 80, 160, 320])])
        self._check_box(ds, 0, 200, 400, [40, 80, 160, 320])
        tmpdir.cleanup()

    def test_box_coords_square(self):
        """Square image, no padding, box coords scale linearly."""
        img = _make_image(224, 224, 2)
        ds, tmpdir = self._make_dataset([(img, [50, 50, 174, 174])])
        self._check_box(ds, 0, 224, 224, [50, 50, 174, 174])
        tmpdir.cleanup()

    def test_box_full_image(self):
        """Full-image box [0,0,w,h] → should become [0,0,1,1]."""
        img = _make_image(400, 300, 3)
        ds, tmpdir = self._make_dataset([(img, [0, 0, 400, 300])])
        _, box_t = ds[0]
        expected = [0.0, 0.0, 1.0, 1.0]
        for got, exp in zip(box_t.tolist(), expected):
            self.assertAlmostEqual(got, exp, delta=0.01, msg=f"Full-box coords off")
        tmpdir.cleanup()

    def test_box_tiny_image(self):
        """Very small 50×100, extreme padding, box still correct."""
        img = _make_image(50, 100, 4)
        ds, tmpdir = self._make_dataset([(img, [10, 20, 40, 80])])
        self._check_box(ds, 0, 50, 100, [10, 20, 40, 80])
        tmpdir.cleanup()

    def test_box_edge_cases(self):
        """Box at various edges: top-left corner, bottom-right corner, narrow strip."""
        # 300×200 landscape → scale = 224/300, pad_x=0, pad_y=(224-149)//2=37
        img = _make_image(300, 200, 5)
        cases = [
            ([0, 0, 50, 50], 300, 200),           # top-left corner
            ([250, 150, 299, 199], 300, 200),      # bottom-right corner
            ([100, 0, 200, 199], 300, 200),        # full-height strip
        ]
        ds, tmpdir = self._make_dataset([(img, box) for box, _, _ in cases])
        for i, (box, w, h) in enumerate(cases):
            self._check_box(ds, i, w, h, box, atol=0.01)
        tmpdir.cleanup()

    # ==============================================================
    # 4. 缺 JSON / 缺图片 边界情况
    # ==============================================================

    def test_missing_json_skip(self):
        """Image without a JSON pair should be silently skipped."""
        tmpdir = tempfile.TemporaryDirectory()
        img1 = _make_image(100, 100, 0)
        img2 = _make_image(100, 100, 1)
        img1.save(os.path.join(tmpdir.name, "0.jpg"))
        img2.save(os.path.join(tmpdir.name, "1.jpg"))
        with open(os.path.join(tmpdir.name, "0.json"), "w") as f:
            json.dump(_make_labelme_json(10, 10, 90, 90), f)
        # No 1.json

        ds = LetterboxCompositionDataset(tmpdir.name)
        self.assertEqual(len(ds), 1, "Should skip 1.jpg without json")
        tmpdir.cleanup()

    def test_no_valid_pairs(self):
        """Directory with no valid image+JSON pairs raises ValueError."""
        tmpdir = tempfile.TemporaryDirectory()
        img = _make_image(100, 100, 0)
        img.save(os.path.join(tmpdir.name, "0.jpg"))
        # No JSON at all
        with self.assertRaises(ValueError):
            LetterboxCompositionDataset(tmpdir.name)
        tmpdir.cleanup()

    def test_custom_target_size(self):
        """Custom target_size (e.g. 128) should produce (3,128,128)."""
        img = _make_image(400, 300, 0)
        ds, tmpdir = self._make_dataset([(img, [50, 40, 350, 260])], target_size=128)
        img_t, box_t = ds[0]
        self.assertEqual(img_t.shape, (3, 128, 128))
        tmpdir.cleanup()

    # ==============================================================
    # 5. 多张图片批量正确性
    # ==============================================================

    def test_mixed_aspect_ratios(self):
        """All images in a mixed batch produce correct output."""
        images = [
            (_make_image(600, 400, 10), [100, 80, 500, 320]),   # landscape
            (_make_image(300, 600, 11), [50, 100, 250, 500]),   # portrait
            (_make_image(224, 224, 12), [20, 20, 204, 204]),    # square
        ]
        ds, tmpdir = self._make_dataset(images)
        for i in range(3):
            img_t, _ = ds[i]
            self.assertEqual(img_t.shape, (3, TARGET, TARGET),
                             f"Mixed batch sample {i}: wrong shape")
        tmpdir.cleanup()


if __name__ == "__main__":
    unittest.main()
