from __future__ import annotations

import ast
import copy
import json
import math
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "TextSwap_Studio.ipynb"
FONT_CANDIDATES = (
    os.environ.get("TEXTSWAP_TEST_FONT"),
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
)
FONT_PATH = next((str(Path(p)) for p in FONT_CANDIDATES if p and Path(p).is_file()), None)


def load_notebook_engine():
    """Load definitions without executing notebook bootstrap, installers, or UI."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(c for c in notebook["cells"] if c.get("id") == "textswap-studio-v2")
    source = "".join(cell["source"])
    tree = ast.parse(source, filename=str(NOTEBOOK))
    stop = next(n.end_lineno for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "edit_page")
    selected = []

    def safe_assignment(node):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return (
            all(isinstance(t, ast.Name) for t in targets)
            and node.value is not None
            and not any(isinstance(n, (ast.Call, ast.Await, ast.Yield)) for n in ast.walk(node.value))
        )

    for node in tree.body:
        if node.lineno > stop:
            break
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            selected.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and safe_assignment(node):
            selected.append(node)
        elif isinstance(node, ast.Try) and "import cv2" in ast.get_source_segment(source, node):
            selected.append(node)
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    loaded = types.ModuleType("textswap_notebook_under_test")
    sys.modules[loaded.__name__] = loaded
    exec(compile(module, str(NOTEBOOK), "exec"), loaded.__dict__)
    # Exercise the real sequential UI edit path without importing the UI or
    # executing notebook bootstrap/installer statements.
    ui_helpers = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_hex_to_bgr", "_apply_edits"}
    ]
    ui_module = ast.Module(body=ui_helpers, type_ignores=[])
    ast.fix_missing_locations(ui_module)
    exec(compile(ui_module, str(NOTEBOOK), "exec"), loaded.__dict__)
    return loaded.__dict__


TS = load_notebook_engine()


def setUpModule():
    print(f"TextSwap renderer test font: {FONT_PATH or 'UNAVAILABLE'}")


def ink_iou(a, b):
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / max(1, union))


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def crop(mask):
    box = bbox(mask)
    return mask if box is None else mask[box[1]:box[3], box[0]:box[2]]


def padded_iou(a, b):
    """Compare cropped natural glyph geometry without rescaling either mask."""
    a, b = crop(a), crop(b)
    h, w = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    aa = np.zeros((h, w), bool)
    bb = np.zeros((h, w), bool)
    aa[:a.shape[0], :a.shape[1]] = a
    bb[:b.shape[0], :b.shape[1]] = b
    return ink_iou(aa, bb)


def changed_mask(before, after, *, padding=0):
    changed = np.any(before != after, axis=2)
    if padding:
        size = 2 * padding + 1
        changed = cv2.dilate(changed.astype(np.uint8), np.ones((size, size), np.uint8)) > 0
    return changed


class RendererTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        if FONT_PATH is None:
            self.skipTest("No supported installed font; set TEXTSWAP_TEST_FONT to a readable TTF/OTF path")
        self.font = FONT_PATH
        self.color = (24.0, 72.0, 164.0)

    def fixture(self, text="TextSwap", *, angle=0.0, gradient=False, size=52):
        h, w = 240, 560
        yy, xx = np.mgrid[:h, :w]
        if gradient:
            background = np.empty((h, w, 3), np.uint8)
            background[..., 0] = np.clip(248 - xx * 0.025 + yy * 0.018, 0, 255)
            background[..., 1] = np.clip(244 - xx * 0.018 + yy * 0.025, 0, 255)
            background[..., 2] = np.clip(236 - xx * 0.012 + yy * 0.030, 0, 255)
        else:
            background = np.full((h, w, 3), (244, 240, 232), np.uint8)
        alpha, L, T = TS["render_alpha"](text, self.font, size, ss=4)
        origin_x, baseline = 82.0, 126.0
        px, py = int(round(origin_x + L)), int(round(baseline + T))
        if abs(angle) >= 0.3:
            local_y, local_x = np.nonzero(alpha > 0.5)
            C = np.array([px + local_x.mean(), py + local_y.mean()])
            th = math.radians(angle)
            R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
            ah, aw = alpha.shape
            t = R @ np.array([px - C[0], py - C[1]]) + C
            corners = np.array([[0, 0], [aw, 0], [0, ah], [aw, ah]], float) @ R.T + t
            mn = np.floor(corners.min(0)).astype(int) - 1
            mx = np.ceil(corners.max(0)).astype(int) + 1
            M = np.hstack([R, (t - mn).reshape(2, 1)]).astype(np.float32)
            alpha = cv2.warpAffine(alpha, M, (int(mx[0] - mn[0]), int(mx[1] - mn[1])), flags=cv2.INTER_LINEAR)
            px, py = int(mn[0]), int(mn[1])
        source_alpha = np.zeros((h, w), np.float32)
        source_alpha[py:py + alpha.shape[0], px:px + alpha.shape[1]] = alpha
        image = background.astype(np.float32)
        image = image * (1 - source_alpha[..., None]) + np.array(self.color) * source_alpha[..., None]
        image = np.clip(image, 0, 255).astype(np.uint8)
        source_mask = source_alpha > 0.5
        box = bbox(source_mask)
        match = TS["Match"](0, text, tuple(box), None, 1.0, text)
        return {
            "text": text,
            "image": image,
            "background": background,
            "alpha": source_alpha,
            "mask": source_mask,
            "box": box,
            "match": match,
            "baseline": baseline,
            "angle": angle,
        }

    def two_block_fixture(self, *, texts=("Alpha", "Beta"), sizes=(52, 52)):
        """Two nearby blocks on one page, each with its own source reference."""
        h, w = 300, 1400
        background = np.full((h, w, 3), (244, 240, 232), np.uint8)
        image = background.astype(np.float32)
        fixtures = []
        for text, size, (origin_x, baseline) in zip(texts, sizes, ((60, 140), (280, 140))):
            alpha, L, T = TS["render_alpha"](text, self.font, size, ss=4)
            px, py = int(round(origin_x + L)), int(round(baseline + T))
            source_alpha = np.zeros((h, w), np.float32)
            source_alpha[py:py + alpha.shape[0], px:px + alpha.shape[1]] = alpha
            image = image * (1 - source_alpha[..., None]) + np.array(self.color) * source_alpha[..., None]
            source_mask = source_alpha > 0.5
            box = bbox(source_mask)
            fixtures.append({
                "text": text,
                "background": background,
                "mask": source_mask,
                "box": box,
                "match": TS["Match"](0, text, tuple(box), None, 1.0, text),
                "baseline": float(baseline),
            })
        image = np.clip(image, 0, 255).astype(np.uint8)
        for fixture in fixtures:
            fixture["image"] = image
        return {"image": image, "fixtures": fixtures}

    def edit_match(self, image, match, replacement, **kwargs):
        info = {}
        options = dict(
            method="fill",
            fonts=[self.font],
            font_override=self.font,
            watermark_mode="off",
            soften=0.0,
            case_match=False,
            style_info=info,
            log=lambda _message: None,
        )
        options.update(kwargs)
        out = TS["edit_page"](TS["Page"](image.copy()), [match], replacement, **options)
        return out, info

    def edit_matches(self, image, matches, replacement, **kwargs):
        """Capture each match's telemetry while using one real edit_page call."""
        captured = []
        original_draw = TS["draw_replacement"]

        def capture_draw(out, reg, old_text, new_text, *args, **draw_kwargs):
            info = {}
            draw_kwargs["style_info"] = info
            result = original_draw(out, reg, old_text, new_text, *args, **draw_kwargs)
            captured.append((old_text, info))
            return result

        options = dict(
            method="fill",
            fonts=[self.font],
            font_override=self.font,
            watermark_mode="off",
            soften=0.0,
            case_match=False,
            weight_match=False,
            log=lambda _message: None,
        )
        options.update(kwargs)
        with mock.patch.dict(TS, {"draw_replacement": capture_draw}):
            out = TS["edit_page"](TS["Page"](image.copy()), matches, replacement, **options)
        return out, dict(captured)

    def apply_sequential(self, fixture, replacements, order):
        """Run ordered pairs through the production _apply_edits function."""
        selected = [
            {"match": fixture["fixtures"][i]["match"], "find": fixture["fixtures"][i]["text"],
             "replace": replacements[i]}
            for i in order
        ]
        settings = {
            "method": "fill", "align": "left", "shrink": False, "auto_skew": False,
            "watermark_mode": "off", "watermark_sensitivity": 0.5,
            "soften_auto": False, "soften": 0.0, "font_override": self.font,
            "color": None, "size_scale": 1.0, "preserve_size": True,
            "case_match": False, "weight_match": False, "stretch_match": True,
            "nudge_x": 0.0, "nudge_y": 0.0, "pixel_lock": True,
        }
        notes = []
        with mock.patch.dict(TS, {"discover_fonts": lambda _root: [self.font]}):
            edited, _logs, styles = TS["_apply_edits"](
                [TS["Page"](fixture["image"].copy())], [[]], selected, settings, notes,
            )
        self.assertEqual(notes, [])
        self.assertEqual(len(styles), 2)
        return edited[0].bgr, {style["find"]: style for style in styles}

    def assert_two_blocks_match_standalone(self, fixture, replacements, actual, actual_styles, standalone):
        """Compare each block only where the other block has no permitted edits."""
        for i, block in enumerate(fixture["fixtures"]):
            expected, expected_info = standalone[i]
            actual_info = actual_styles[block["text"]]
            label = f"block={block['text']}"
            # The direct standalone edit is this block's natural whole-line
            # reference; exact style/bounds/baseline comparisons avoid using
            # the first block as a shared style oracle.
            for key in ("size", "stretch", "weight", "soften", "shift"):
                self.assertEqual(actual_info[key], expected_info[key], f"{label}: {key}")
            for key in (
                "source_core_bbox", "source_reconstruction_bbox", "source_reconstruction_iou",
                "replacement_ink_bbox_derotated", "replacement_ink_bbox",
                "baseline_y_derotated", "calibration_shift",
            ):
                self.assertEqual(actual_info["pixel"][key], expected_info["pixel"][key], f"{label}: {key}")
            expected_box = expected_info["pixel"]["replacement_ink_bbox"]
            actual_box = actual_info["pixel"]["replacement_ink_bbox"]
            expected_height = expected_box[3] - expected_box[1]
            actual_height = actual_box[3] - actual_box[1]
            self.assertEqual(actual_height, expected_height, f"{label}: ink height")
            self.assertEqual(
                actual_info["pixel"]["baseline_y_derotated"],
                expected_info["pixel"]["baseline_y_derotated"],
                f"{label}: baseline",
            )
            # Three pixels cover the antialiased overlap envelope. Same-colour
            # source ink can hide a legal replacement pixel in a changed-mask,
            # so two-pixel clipping is not a sufficient ownership boundary.
            own_edit = changed_mask(fixture["image"], expected, padding=3)
            other_edit = changed_mask(fixture["image"], standalone[1 - i][0], padding=3)
            exclusive_own_edit = own_edit & ~other_edit
            self.assertTrue(exclusive_own_edit.any(), f"{label}: empty owned edit region")
            self.assertTrue(
                np.array_equal(actual[exclusive_own_edit], expected[exclusive_own_edit]),
                f"{label}: sequential/combined pixels differ from its standalone whole-line reference",
            )

    def edit(self, fixture, replacement=Ellipsis, **kwargs):
        replacement = fixture["text"] if replacement is Ellipsis else replacement
        return self.edit_match(fixture["image"], fixture["match"], replacement, **kwargs)

    def coverage(self, fixture, image):
        background = fixture["background"].astype(np.float32)
        observed = image.astype(np.float32)
        delta = background - np.array(self.color, np.float32)
        denominator = np.maximum(1.0, np.sum(delta * delta, axis=2))
        return np.sum((background - observed) * delta, axis=2) / denominator

    def assert_identical_reconstruction(self, fixture, out, info, *, check_baseline=True, check_source_iou=True):
        near = cv2.dilate(fixture["mask"].astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        rendered = (self.coverage(fixture, out) > 0.5) & near
        iou = ink_iou(rendered, fixture["mask"])
        pixel = info["pixel"]
        json.dumps(info)
        self.assertGreaterEqual(iou, 0.95, f"identical-text ink IoU={iou:.4f}; telemetry={pixel}")
        self.assertIsNotNone(pixel["source_reconstruction_iou"])
        if check_source_iou:
            self.assertGreaterEqual(
                pixel["source_reconstruction_iou"], 0.95,
                f"source reconstruction IoU={pixel['source_reconstruction_iou']}; output IoU={iou:.4f}",
            )
        if check_baseline:
            error = abs(pixel["baseline_y_derotated"] - fixture["baseline"])
            self.assertLessEqual(error, 1.0, f"baseline error={error}; telemetry={pixel}")
        outside = ~cv2.dilate(fixture["mask"].astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
        self.assertTrue(np.array_equal(out[outside], fixture["image"][outside]), "pixels outside the edit changed")
        self.assertIn(pixel["method"], ("whole-line-calibrated", "whole-line"))
        self.assertEqual(pixel["tracking"], 0.0)
        self.assertEqual(pixel["replacement_scale_y"], 1.0)
        for key in ("chars_ok", "height_exact", "char_count", "char_height_exact"):
            self.assertNotIn(key, pixel)
        self.assertNotIn("chars", info)
        self.assertNotIn("char_map", info)
        return iou

    def assert_natural_geometry(self, fixture, replacement, out, info):
        pixel = info["pixel"]
        actual_box = pixel["replacement_ink_bbox_derotated"]
        self.assertIsNotNone(actual_box)
        near = np.zeros(out.shape[:2], bool)
        near[actual_box[1]:actual_box[3], actual_box[0]:actual_box[2]] = True
        actual = crop((self.coverage(fixture, out) > 0.5) & near)
        reference, _, ref_T = TS["render_alpha"](replacement, self.font, info["size"], ss=4)
        scale = info["stretch"]
        if abs(scale - 1.0) > 0.002:
            reference = cv2.resize(reference, (max(1, round(reference.shape[1] * scale)), reference.shape[0]), interpolation=cv2.INTER_AREA)
        reference = reference > 0.5
        ref_box = bbox(reference)
        ref_h, ref_w = ref_box[3] - ref_box[1], ref_box[2] - ref_box[0]
        self.assertLessEqual(abs(actual.shape[0] - ref_h), 1, f"natural height differs: {actual.shape} vs {(ref_h, ref_w)}")
        self.assertLessEqual(abs(actual.shape[1] - ref_w), 1, f"natural width differs: {actual.shape} vs {(ref_h, ref_w)}")
        iou = padded_iou(actual, reference)
        self.assertGreaterEqual(iou, 0.95, f"whole-line natural-geometry IoU={iou:.4f}")
        baseline_error = abs(pixel["baseline_y_derotated"] - fixture["baseline"])
        self.assertLessEqual(baseline_error, 1.0)
        expected_descent = ref_T + ref_box[3]
        actual_descent = actual_box[3] - pixel["baseline_y_derotated"]
        self.assertLessEqual(abs(actual_descent - expected_descent), 1.0)

    def test_identical_text_flat_background(self):
        fixture = self.fixture()
        out, info = self.edit(fixture, font_override=None)
        self.assert_identical_reconstruction(fixture, out, info)

    def test_identical_text_gradient_background_and_color(self):
        fixture = self.fixture(gradient=True)
        out, info = self.edit(fixture)
        self.assert_identical_reconstruction(fixture, out, info)
        strong = fixture["mask"] & (self.coverage(fixture, out) > 0.9)
        self.assertGreater(strong.sum(), 20)
        mean_error = np.abs(out[strong].astype(float) - np.array(self.color)).mean()
        self.assertLessEqual(mean_error, 8.0)
        self.assertEqual(info["color"], "#a44818")

    def test_identical_text_pixel_lock_disabled_still_whole_line(self):
        fixture = self.fixture()
        out, info = self.edit(fixture, pixel_lock=False)
        iou = self.assert_identical_reconstruction(fixture, out, info)
        self.assertGreaterEqual(iou, 0.95)
        self.assertEqual(info["pixel"]["method"], "whole-line")

    def test_rotation_positive_and_negative(self):
        for angle in (4.0, -4.0):
            with self.subTest(angle=angle):
                fixture = self.fixture(angle=angle)
                out, info = self.edit(fixture, skew=angle)
                self.assertEqual(info["pixel"]["rotation"], angle)
                # Final page ink is the controlled ground truth.  The nearest-
                # neighbour de-rotated OCR core is lossy, so its diagnostic IoU
                # is reported but is not used to relax the final 0.95 gate.
                self.assert_identical_reconstruction(
                    fixture, out, info, check_baseline=False, check_source_iou=False,
                )

    def test_different_lengths_preserve_natural_whole_line_geometry(self):
        for old, new in (("W", "iiiii"), ("iiiii", "W"), ("MMMM", "iiii")):
            with self.subTest(old=old, new=new):
                fixture = self.fixture(old)
                out, info = self.edit(fixture, new, weight_match=False)
                self.assert_natural_geometry(fixture, new, out, info)

    def test_two_blocks_multiple_matches_are_order_independent(self):
        for sizes in ((52, 52), (72, 36)):
            for replacement in ("MMMM", "ii"):
                for order in ((0, 1), (1, 0)):
                    with self.subTest(path="combined", sizes=sizes, replacement=replacement, order=order):
                        fixture = self.two_block_fixture(texts=("W", "iiiii"), sizes=sizes)
                        blocks = fixture["fixtures"]
                        matches = [blocks[i]["match"] for i in order]
                        actual, actual_styles = self.edit_matches(fixture["image"], matches, replacement)
                        standalone = [
                            self.edit_match(
                                fixture["image"], block["match"], replacement, weight_match=False,
                            )
                            for block in blocks
                        ]
                        self.assert_two_blocks_match_standalone(
                            fixture, (replacement, replacement), actual, actual_styles, standalone,
                        )

    def test_two_blocks_sequential_apply_edits_are_order_independent(self):
        cases = (
            ((52, 52), ("LONG FIRST REPLACEMENT", "W")),
            ((52, 52), ("MM", "WW")),
            ((72, 36), ("LONG FIRST REPLACEMENT", "W")),
            ((72, 36), ("MM", "WW")),
        )
        for sizes, replacements in cases:
            fixture = self.two_block_fixture(sizes=sizes)
            standalone = [
                self.edit_match(
                    fixture["image"], block["match"], replacements[i], weight_match=False,
                )
                for i, block in enumerate(fixture["fixtures"])
            ]
            for order in ((0, 1), (1, 0)):
                with self.subTest(path="sequential", sizes=sizes, replacements=replacements, order=order):
                    actual, actual_styles = self.apply_sequential(fixture, replacements, order)
                    self.assert_two_blocks_match_standalone(
                        fixture, replacements, actual, actual_styles, standalone,
                    )

    def test_caps_and_descenders_share_the_source_baseline(self):
        for old, new in (("CAPS", "gypq"), ("gypq", "CAPS")):
            with self.subTest(old=old, new=new):
                fixture = self.fixture(old)
                out, info = self.edit(fixture, new, weight_match=False)
                self.assert_natural_geometry(fixture, new, out, info)

    def test_nudge_alignment_case_and_overrides(self):
        fixture = self.fixture("SOURCE")
        plain, plain_info = self.edit(fixture, "source", case_match=True, weight_match=False)
        self.assertEqual(plain_info["applied"], "SOURCE")
        nudged, nudge_info = self.edit(
            fixture, "source", case_match=True, weight_match=False, nudge_x=3, nudge_y=2,
        )
        self.assertEqual(nudge_info["applied"], "SOURCE")
        a = plain_info["pixel"]["replacement_ink_bbox"]
        b = nudge_info["pixel"]["replacement_ink_bbox"]
        self.assertLessEqual(abs((b[0] - a[0]) - 3), 1)
        self.assertLessEqual(abs((b[1] - a[1]) - 2), 1)
        self.assertEqual(nudge_info["pixel"]["nudge"], [3.0, 2.0])
        for align in ("center", "right"):
            with self.subTest(align=align):
                out, info = self.edit(fixture, "ii", align=align, weight_match=False)
                box = info["pixel"]["replacement_ink_bbox"]
                shift = info["pixel"]["calibration_shift"][0]
                if align == "center":
                    actual = (box[0] + box[2]) / 2
                    expected = (fixture["box"][0] + fixture["box"][2]) / 2 + shift
                else:
                    actual = box[2]
                    expected = fixture["box"][2] + shift
                self.assertLessEqual(abs(actual - expected), 1.0)

    def test_shrink_to_gap_preserves_whole_line_geometry(self):
        fixture = self.fixture("W")
        fixture["match"].next_x = fixture["box"][2] + 50
        out, info = self.edit(
            fixture, "WWWWWWWW", shrink=True, preserve_size=False,
            weight_match=False,
        )
        self.assertTrue(info["size"] < 52.0 or info["stretch"] < 1.0)
        self.assert_natural_geometry(fixture, "WWWWWWWW", out, info)

    def test_softness_weight_and_color_controls(self):
        fixture = self.fixture("Control")
        out, info = self.edit(fixture, "Control", soften=0.8, weight_match=False, color=(10, 20, 200))
        self.assertEqual(info["soften"], 0.8)
        self.assertEqual(info["weight"], 0.0)
        self.assertEqual(info["color"], "#c8140a")
        self.assertIsNotNone(info["pixel"]["replacement_ink_bbox"])

    def test_no_replacement_erases_without_drawing(self):
        fixture = self.fixture("Erase me")
        out, info = self.edit(fixture, None)
        remaining = self.coverage(fixture, out)[fixture["mask"]]
        self.assertLess(float(np.mean(remaining > 0.5)), 0.05)
        self.assertNotIn("pixel", info)
        outside = ~cv2.dilate(fixture["mask"].astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
        self.assertTrue(np.array_equal(out[outside], fixture["image"][outside]))

    def test_malformed_ocr_box_and_no_ink_are_safe(self):
        blank = np.full((120, 220, 3), 255, np.uint8)
        for box in ((40, 30, 90, 70), (40, 40, 40, 80), (-20, -20, -5, -5)):
            with self.subTest(box=box):
                match = TS["Match"](0, "ghost", box, None, 1.0, "ghost")
                messages = []
                out = TS["edit_page"](
                    TS["Page"](blank), [match], "replacement", fonts=[self.font],
                    watermark_mode="off", log=messages.append,
                )
                self.assertTrue(np.array_equal(out, blank))
                self.assertTrue(any("no ink" in message for message in messages))

    def test_notebook_has_no_slot_snap_telemetry(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cell = next(c for c in notebook["cells"] if c.get("id") == "textswap-studio-v2")
        source = "".join(cell["source"])
        self.assertNotIn("p.chars_ok", source)
        self.assertNotIn('"method": "chars"', source)
        self.assertIn("source_reconstruction_iou", source)
        self.assertIn("replacement_ink_bbox", source)
        self.assertIn("baseline_offset", source)

    def test_missing_glyph_fails_explicitly(self):
        fixture = self.fixture("Missing")
        with self.assertRaisesRegex(RuntimeError, "no available font has all the needed characters"):
            self.edit(fixture, "\uffff")

    def test_over_watermark_uses_common_compositor(self):
        fixture = self.fixture("Veil")
        reg = TS["analyze_region"](fixture["image"], fixture["match"].box, watermark_mode="off")
        self.assertIsNotNone(reg)
        erased = fixture["image"].copy()
        TS["erase"](erased, fixture["image"], reg, "fill")
        plain = erased.copy()
        TS["draw_replacement"](
            plain, reg, fixture["text"], fixture["text"], [self.font],
            font_override=self.font, soften=0.0, weight_match=False, case_match=False,
        )
        veiled_reg = copy.deepcopy(reg)
        h, w = veiled_reg.core.shape
        veiled_reg.underlay = TS["UnderlayProfile"](
            present=True, order="over", confidence=1.0,
            alpha=np.full((h, w), 0.5, np.float32), color=(255.0, 255.0, 255.0),
        )
        veiled = erased.copy()
        TS["draw_replacement"](
            veiled, veiled_reg, fixture["text"], fixture["text"], [self.font],
            font_override=self.font, soften=0.0, weight_match=False, case_match=False,
        )
        changed = np.any(plain != erased, axis=2)
        self.assertGreater(changed.sum(), 20)
        plain_lum = cv2.cvtColor(plain, cv2.COLOR_BGR2GRAY)[changed].mean()
        veiled_lum = cv2.cvtColor(veiled, cv2.COLOR_BGR2GRAY)[changed].mean()
        self.assertGreater(veiled_lum, plain_lum + 5.0)


if __name__ == "__main__":
    unittest.main()
