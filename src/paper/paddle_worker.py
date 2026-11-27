#!/usr/bin/env python
"""PaddleOCR worker for paper.readers.PaddleReader; runs in its own conda env.

stdin: one JSON line per request {"path": png}.  Original stdout: one JSON line per
request {"items": [[y_centre, x_centre, height, text], ...]} or {"error": "..."}.
Library output is routed to stderr so it cannot corrupt the protocol.
"""
import json
import os
import sys

proto = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr


def main():
    import numpy as np
    from PIL import Image
    import paddle
    import paddleocr
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False, device="cpu", enable_mkldnn=False)
    proto.write(json.dumps({"ready": True, "paddleocr": paddleocr.__version__,
                            "paddlepaddle": paddle.__version__}) + "\n")
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        if not line.strip():
            continue
        try:
            path = json.loads(line)["path"]
            a = np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()
            items = []
            for res in ocr.predict(a):
                for t, poly in zip(res["rec_texts"], res["rec_polys"]):
                    q = np.asarray(poly, dtype=float)
                    items.append([float(q[:, 1].mean()), float(q[:, 0].mean()), float(np.ptp(q[:, 1])), str(t)])
            reply = {"items": items}
        except Exception as exc:
            reply = {"error": repr(exc)}
        proto.write(json.dumps(reply) + "\n")


if __name__ == "__main__":
    main()
